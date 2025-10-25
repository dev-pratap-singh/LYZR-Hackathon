"""
Database connection and initialization with support for Azure managed services
"""
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool, QueuePool
from contextlib import contextmanager
from typing import Generator
import logging

from app.config import settings
from app.models import Base

logger = logging.getLogger(__name__)


def _create_postgres_engine():
    """
    Create PostgreSQL engine with appropriate pooling strategy
    based on deployment environment
    """
    pool_settings = settings.get_database_connection_pool_settings()

    if settings.is_azure_deployment():
        # Use connection pooling for Azure (production)
        logger.info(f"Creating PostgreSQL engine for Azure with connection pooling: {pool_settings}")
        engine = create_engine(
            settings.postgres_url,
            poolclass=QueuePool,
            pool_size=pool_settings["pool_size"],
            max_overflow=pool_settings["max_overflow"],
            pool_timeout=pool_settings["pool_timeout"],
            pool_recycle=pool_settings["pool_recycle"],
            pool_pre_ping=pool_settings["pool_pre_ping"],
            echo=False,
            connect_args={
                "connect_timeout": 10,
                "options": "-c statement_timeout=30000"  # 30 second query timeout
            }
        )
    else:
        # Use NullPool for local development (simpler for testing)
        logger.info("Creating PostgreSQL engine for local development (NullPool)")
        engine = create_engine(
            settings.postgres_url,
            poolclass=NullPool,
            echo=False
        )

    return engine


def _create_pgvector_engine():
    """
    Create PGVector engine with appropriate pooling strategy
    based on deployment environment
    """
    pool_settings = settings.get_database_connection_pool_settings()

    if settings.is_azure_deployment():
        # Use connection pooling for Azure (production)
        logger.info(f"Creating PGVector engine for Azure with connection pooling: {pool_settings}")
        engine = create_engine(
            settings.pgvector_url,
            poolclass=QueuePool,
            pool_size=pool_settings["pool_size"],
            max_overflow=pool_settings["max_overflow"],
            pool_timeout=pool_settings["pool_timeout"],
            pool_recycle=pool_settings["pool_recycle"],
            pool_pre_ping=pool_settings["pool_pre_ping"],
            echo=False,
            connect_args={
                "connect_timeout": 10,
                "options": "-c statement_timeout=30000"  # 30 second query timeout
            }
        )
    else:
        # Use NullPool for local development
        logger.info("Creating PGVector engine for local development (NullPool)")
        engine = create_engine(
            settings.pgvector_url,
            poolclass=NullPool,
            echo=False
        )

    return engine


# PostgreSQL engine for document metadata
postgres_engine = _create_postgres_engine()

# PGVector engine for embeddings
pgvector_engine = _create_pgvector_engine()

# Session makers
PostgresSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=postgres_engine)
PGVectorSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=pgvector_engine)


def init_databases():
    """Initialize both databases with required extensions and tables"""
    try:
        # Initialize PostgreSQL
        logger.info("Initializing PostgreSQL database...")
        Base.metadata.create_all(bind=postgres_engine, tables=[
            Base.metadata.tables['documents']
        ])
        logger.info("✓ PostgreSQL database initialized")

        # Initialize Memory Management tables (custom schema)
        logger.info("Initializing Memory Management schema...")
        _init_memory_schema()
        logger.info("✓ Memory Management schema initialized")

        # Initialize Memory Repo tables (from memory/src)
        logger.info("Initializing Memory Repo tables...")
        _init_memory_repo_tables()
        logger.info("✓ Memory Repo tables initialized")

        # Initialize PGVector with vector extension
        logger.info("Initializing PGVector database...")
        with pgvector_engine.connect() as conn:
            # Enable pgvector extension
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()

        # Create embeddings table
        Base.metadata.create_all(bind=pgvector_engine, tables=[
            Base.metadata.tables['embeddings']
        ])

        # Create vector index for faster similarity search
        with pgvector_engine.connect() as conn:
            try:
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS embeddings_vector_idx
                    ON embeddings
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                """))
                conn.commit()
                logger.info("✓ Vector index created")
            except Exception as e:
                logger.warning(f"Vector index creation skipped: {e}")

        logger.info("✓ PGVector database initialized")
        return True

    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


def _init_memory_repo_tables():
    """Initialize memory repo tables from memory/src/database/models.py

    NOTE: Memory repo tables are created in the PGVECTOR database because they use
    Vector columns for embeddings. The pgvector/pgvector:pg15 image has the vector extension.
    """
    import sys
    import os

    try:
        # Add memory directory to Python path
        # Memory src is mounted at /app/memory/src in the container
        memory_path = '/app/memory'
        if memory_path not in sys.path:
            sys.path.insert(0, memory_path)

        # Import memory repo models
        from src.database.models import Base as MemoryBase

        # Create tables using memory repo's Base in PGVECTOR database
        # (pgvector extension is already enabled in the pgvector container)
        MemoryBase.metadata.create_all(bind=pgvector_engine)

        # Verify tables were created
        import psycopg2
        conn = psycopg2.connect(
            host=settings.pgvector_host,
            port=settings.pgvector_port,
            database=settings.pgvector_db,
            user=settings.postgres_user,
            password=settings.postgres_password
        )

        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name IN ('conversations', 'memory_facts', 'user_preferences', 'training_history')
            """)
            tables = [row[0] for row in cursor.fetchall()]
            cursor.close()

            if len(tables) >= 2:
                logger.info(f"✓ Verified memory repo tables: {', '.join(tables)}")
            else:
                logger.warning(f"⚠️ Only found {len(tables)} memory repo tables: {', '.join(tables)}")

        finally:
            conn.close()

    except Exception as e:
        logger.error(f"Error creating memory repo tables: {e}")
        import traceback
        traceback.print_exc()
        # Don't raise - allow app to continue even if memory repo tables fail
        logger.warning("Continuing without memory repo tables")


def _init_memory_schema():
    """Initialize memory management schema with all tables and indexes"""
    import os
    import psycopg2

    try:
        # Get the path to the SQL schema file
        schema_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'init_memory_schema.sql'
        )

        # Read the SQL file
        with open(schema_file, 'r') as f:
            sql_content = f.read()

        # Create the update_updated_at_column function first (required for triggers)
        with postgres_engine.connect() as conn:
            conn.execute(text("""
                CREATE OR REPLACE FUNCTION update_updated_at_column()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.updated_at = CURRENT_TIMESTAMP;
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;
            """))
            conn.commit()
            logger.info("✓ Created update_updated_at_column function")

        # Use psycopg2 directly to execute the entire SQL file
        # SQLAlchemy's text() has issues with complex SQL scripts
        conn = psycopg2.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            database=settings.postgres_db,
            user=settings.postgres_user,
            password=settings.postgres_password
        )

        try:
            cursor = conn.cursor()
            # Execute the entire SQL file at once
            cursor.execute(sql_content)
            conn.commit()
            cursor.close()
            logger.info("✓ Memory management tables and indexes created")

            # Verify tables were created
            cursor = conn.cursor()
            cursor.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name IN ('memory_items', 'memory_state', 'tasks', 'agents')
            """)
            tables = [row[0] for row in cursor.fetchall()]
            cursor.close()

            if len(tables) >= 2:
                logger.info(f"✓ Verified memory tables: {', '.join(tables)}")
            else:
                logger.warning(f"⚠️ Only found {len(tables)} memory tables: {', '.join(tables)}")

        finally:
            conn.close()

        logger.info("✓ Memory management schema initialization completed")

    except FileNotFoundError:
        logger.warning("Memory schema SQL file not found. Skipping memory table initialization.")
        raise
    except Exception as e:
        logger.error(f"Error initializing memory schema: {e}")
        import traceback
        traceback.print_exc()
        raise


@contextmanager
def get_postgres_db() -> Generator[Session, None, None]:
    """Get PostgreSQL database session"""
    db = PostgresSessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_pgvector_db() -> Generator[Session, None, None]:
    """Get PGVector database session"""
    db = PGVectorSessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_postgres_session():
    """Dependency for FastAPI endpoints"""
    db = PostgresSessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_pgvector_session():
    """Dependency for FastAPI endpoints"""
    db = PGVectorSessionLocal()
    try:
        yield db
    finally:
        db.close()

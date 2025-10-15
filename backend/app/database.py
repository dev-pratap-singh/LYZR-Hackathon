"""
Database connection and initialization
"""
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool
from contextlib import contextmanager
from typing import Generator
import logging

from app.config import settings
from app.models import Base

logger = logging.getLogger(__name__)


# PostgreSQL engine for document metadata
postgres_engine = create_engine(
    settings.postgres_url,
    poolclass=NullPool,
    echo=False
)

# PGVector engine for embeddings
pgvector_engine = create_engine(
    settings.pgvector_url,
    poolclass=NullPool,
    echo=False
)

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

        # Initialize Memory Management tables
        logger.info("Initializing Memory Management schema...")
        _init_memory_schema()
        logger.info("✓ Memory Management schema initialized")

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


def _init_memory_schema():
    """Initialize memory management schema with all tables and indexes"""
    import os

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

        # Execute the memory schema SQL
        with postgres_engine.connect() as conn:
            # Split by semicolon and execute each statement
            statements = sql_content.split(';')
            for statement in statements:
                statement = statement.strip()
                if statement:
                    try:
                        conn.execute(text(statement))
                    except Exception as e:
                        # Log but don't fail on individual statement errors
                        # (e.g., if tables already exist)
                        logger.debug(f"Statement execution note: {e}")

            conn.commit()
            logger.info("✓ Memory management tables and indexes created")

    except FileNotFoundError:
        logger.warning("Memory schema SQL file not found. Skipping memory table initialization.")
    except Exception as e:
        logger.error(f"Error initializing memory schema: {e}")
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

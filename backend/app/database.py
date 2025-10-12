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

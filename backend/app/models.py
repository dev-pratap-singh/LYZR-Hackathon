"""
Database models for RAG System
"""
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
import uuid

Base = declarative_base()


class Document(Base):
    """Document metadata table in PostgreSQL"""
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(255), nullable=False)
    filepath = Column(String(512), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_size = Column(Integer)
    file_type = Column(String(50), default="pdf")

    # Processing status
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())
    processed_at = Column(DateTime(timezone=True), nullable=True)
    is_processed = Column(Boolean, default=False)
    processing_status = Column(String(50), default="pending")  # pending, processing, completed, failed

    # Text extraction
    text_filepath = Column(String(512), nullable=True)  # Path to extracted .txt file
    total_chunks = Column(Integer, default=0)

    # Metadata
    user_id = Column(String(100), default="default_user")
    doc_metadata = Column(JSON, default={})

    # Error handling
    error_message = Column(Text, nullable=True)


class Embedding(Base):
    """Vector embeddings table in PGVector database"""
    __tablename__ = "embeddings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), nullable=False, index=True)

    # Chunk information
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    chunk_size = Column(Integer)

    # Vector embedding (1536 dimensions for text-embedding-3-small)
    embedding = Column(Vector(1536), nullable=False)

    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    chunk_metadata = Column(JSON, default={})

    # For hybrid search
    chunk_hash = Column(String(64), index=True)  # Hash for deduplication

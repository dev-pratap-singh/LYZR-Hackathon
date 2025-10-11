"""
Document Processing Helper Functions
Extracted from API routes for better code organization
"""
import logging
from datetime import datetime
from typing import Literal

from sqlalchemy.orm import Session

from app.models import Document
from app.services.document_processor import DocumentProcessingService
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStoreService
from app.services.bm25_search import BM25SearchService

logger = logging.getLogger(__name__)

ProcessingMethod = Literal["pymupdf", "docling"]


async def process_document_pipeline(
    document: Document,
    db: Session,
    method: ProcessingMethod = "pymupdf"
) -> None:
    """
    Process document: extract text, chunk, embed, store

    Args:
        document: Document model instance
        db: Database session
        method: Processing method to use ("pymupdf" or "docling")

    Raises:
        Exception: If processing fails
    """
    doc_processor = DocumentProcessingService()
    embedding_service = EmbeddingService()
    vector_store = VectorStoreService()
    bm25_service = BM25SearchService()

    try:
        logger.info(f"Processing document: {document.id} with method: {method}")

        # Update status
        document.processing_status = "processing"
        db.commit()

        # Read file content
        with open(document.filepath, 'rb') as f:
            content = f.read()

        # Process with selected method
        result = await doc_processor.process_document(
            content,
            document.original_filename,
            str(document.id),
            method=method
        )

        if not result['success']:
            document.processing_status = "failed"
            document.error_message = result.get('error', 'Unknown error')
            db.commit()
            raise Exception(result.get('error', 'Processing failed'))

        # Update document with text path
        document.text_filepath = result['text_path']
        db.commit()

        # Generate embeddings
        logger.info("Generating embeddings...")
        chunk_texts = [chunk['text'] for chunk in result['chunks']]
        embeddings = await embedding_service.generate_embeddings_batch(chunk_texts)

        # Store in PGVector
        logger.info("Storing embeddings in PGVector...")
        stored_count = await vector_store.store_embeddings(
            document.id,
            result['chunks'],
            embeddings
        )

        # Build BM25 index for keyword search
        logger.info("Building BM25 index for keyword search...")
        await bm25_service.build_index(document.id)
        logger.info(f"✓ Built BM25 index for document {document.id}")

        # Update document status
        document.is_processed = True
        document.processing_status = "completed"
        document.processed_at = datetime.now()
        document.total_chunks = stored_count
        db.commit()

        logger.info(f"✅ Document processed successfully: {document.id} ({method})")
        logger.info(f"   - Chunks: {stored_count}")
        logger.info(f"   - Text length: {result['text_length']} chars")

    except Exception as e:
        logger.error(f"❌ Document processing error: {e}")
        document.processing_status = "failed"
        document.error_message = str(e)
        db.commit()
        raise

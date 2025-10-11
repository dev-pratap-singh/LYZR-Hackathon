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
from app.services.graphrag_pipeline import GraphRAGPipeline
from app.services.neo4j_service import neo4j_service
from app.config import settings

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

        # Update document status (Vector search ready)
        document.is_processed = True
        document.processing_status = "completed"
        document.processed_at = datetime.now()
        document.total_chunks = stored_count
        db.commit()

        logger.info(f"✅ Document processed successfully: {document.id} ({method})")
        logger.info(f"   - Chunks: {stored_count}")
        logger.info(f"   - Text length: {result['text_length']} chars")

        # ====== GraphRAG Processing (if enabled) ======
        if settings.graphrag_enabled:
            try:
                logger.info(f"🕸️ Starting GraphRAG processing for document {document.id}...")

                # Initialize GraphRAG pipeline
                graphrag_pipeline = GraphRAGPipeline()

                # Read extracted text
                with open(document.text_filepath, 'r', encoding='utf-8') as f:
                    text_content = f.read()

                # Process with GraphRAG (extract entities and relationships)
                graph_result = await graphrag_pipeline.process_document(
                    text_content=text_content,
                    document_id=str(document.id)
                )

                logger.info(f"✓ GraphRAG extraction complete: {graph_result['entities_count']} entities, {graph_result['relationships_count']} relationships")

                # Create Neo4j constraints and indexes
                await neo4j_service.create_constraints()
                logger.info("✓ Neo4j constraints created")

                # Import all graph documents into Neo4j (one per chunk)
                logger.info(f"Importing graph into Neo4j ({len(graph_result['graph_documents'])} chunks)...")
                await neo4j_service.import_graph_documents(
                    graph_documents=graph_result["graph_documents"],  # Now a list from all chunks
                    document_id=str(document.id)
                )
                logger.info("✓ Graph imported to Neo4j")

                # Update document with graph statistics
                document.graph_processed = True
                document.graph_entities_count = graph_result["entities_count"]
                document.graph_relationships_count = graph_result["relationships_count"]
                document.graph_processing_time = graph_result["processing_time"]
                db.commit()

                logger.info(f"✅ GraphRAG processing complete for {document.id}")
                logger.info(f"   - Chunks processed: {graph_result['chunks_processed']}")
                logger.info(f"   - Entities: {graph_result['entities_count']}")
                logger.info(f"   - Relationships: {graph_result['relationships_count']}")
                logger.info(f"   - Processing time: {graph_result['processing_time']}s")

            except Exception as graph_error:
                # Log error but don't fail the entire pipeline
                # Vector search will still work
                logger.error(f"⚠️ GraphRAG processing failed (vector search still available): {graph_error}")
                document.graph_processed = False
                document.error_message = f"Graph processing failed: {str(graph_error)}"
                db.commit()
        else:
            logger.info("ℹ️ GraphRAG processing disabled (GRAPHRAG_ENABLED=false)")

    except Exception as e:
        logger.error(f"❌ Document processing error: {e}")
        document.processing_status = "failed"
        document.error_message = str(e)
        db.commit()
        raise

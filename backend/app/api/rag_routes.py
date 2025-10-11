"""
RAG System API Routes
Handles document upload, query processing, and streaming responses
"""
import logging
import os
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Literal

from sqlalchemy.orm import Session

from app.database import get_postgres_session
from app.models import Document
from app.services.search_agent import SearchAgent
from app.services.vector_store import VectorStoreService
from app.services.document_processor import DocumentProcessingService
from app.services.bm25_search import BM25SearchService
from app.services.neo4j_service import neo4j_service
from app.services.graphrag_pipeline import GraphRAGPipeline
from app.utils.document_helpers import process_document_pipeline

logger = logging.getLogger(__name__)

ProcessingMethod = Literal["pymupdf", "docling"]

router = APIRouter(prefix="/api/rag", tags=["RAG System"])

# Initialize services
search_agent = SearchAgent()
vector_store = VectorStoreService()
doc_processor = DocumentProcessingService()
bm25_service = BM25SearchService()


# ==== Pydantic Models ====
class QueryRequest(BaseModel):
    query: str
    document_id: Optional[str] = None


class UploadRequest(BaseModel):
    method: Optional[ProcessingMethod] = "pymupdf"


class DocumentResponse(BaseModel):
    id: str
    filename: str
    uploaded_at: str
    is_processed: bool
    processing_status: str
    total_chunks: int


# ==== Upload Endpoint ====
@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    method: ProcessingMethod = "pymupdf",
    db: Session = Depends(get_postgres_session)
):
    """
    Upload a PDF document and process it

    Args:
        file: PDF file to upload
        method: Processing method ("pymupdf" or "docling")
        db: Database session

    Returns:
        Upload status with document ID and processing info
    """
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        # Read file content
        content = await file.read()

        # Create document record
        document = Document(
            filename=file.filename,
            filepath="",  # Will be updated after saving
            original_filename=file.filename,
            file_size=len(content),
            file_type="pdf",
            processing_status="pending"
        )

        db.add(document)
        db.commit()
        db.refresh(document)

        logger.info(f"Created document record: {document.id}")

        # Save file (update document with filepath)
        file_path, file_hash = await doc_processor.save_uploaded_file(content, file.filename)
        document.filepath = file_path
        db.commit()

        # Process document immediately with selected method
        try:
            await process_document_pipeline(document, db, method=method)
            logger.info(f"Document processed successfully with {method}: {document.id}")
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            # Return success for upload but indicate processing failed
            return {
                "success": True,
                "document_id": str(document.id),
                "filename": file.filename,
                "method": method,
                "message": f"Document uploaded but processing failed: {str(e)}",
                "status": "processing_failed"
            }

        return {
            "success": True,
            "document_id": str(document.id),
            "filename": file.filename,
            "method": method,
            "message": f"Document uploaded and processed successfully with {method}. {document.total_chunks} chunks created.",
            "status": "completed"
        }

    except Exception as e:
        logger.error(f"Upload error: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


# ==== Query Streaming Endpoint ====
@router.post("/query/stream")
async def query_stream(
    request: QueryRequest,
    db: Session = Depends(get_postgres_session)
):
    """
    Process query with streaming response

    - Generates embeddings for query
    - Performs vector search
    - Streams LLM response with reasoning
    """
    try:
        # If document_id provided, verify it exists and process if needed
        document_id = None
        if request.document_id:
            document = db.query(Document).filter(Document.id == request.document_id).first()
            if not document:
                raise HTTPException(status_code=404, detail="Document not found")

            # Process document if not already processed
            if not document.is_processed:
                await process_document_pipeline(document, db)

            document_id = request.document_id

        # Stream query results
        return StreamingResponse(
            search_agent.process_query_with_streaming(request.query, document_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==== List Documents Endpoint ====
@router.get("/documents", response_model=List[DocumentResponse])
async def list_documents(db: Session = Depends(get_postgres_session)):
    """Get list of all uploaded documents"""
    try:
        documents = db.query(Document).order_by(Document.uploaded_at.desc()).all()

        return [{
            "id": str(doc.id),
            "filename": doc.original_filename,
            "uploaded_at": doc.uploaded_at.isoformat(),
            "is_processed": doc.is_processed,
            "processing_status": doc.processing_status,
            "total_chunks": doc.total_chunks
        } for doc in documents]

    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==== Get Document Status ====
@router.get("/documents/{document_id}")
async def get_document(
    document_id: str,
    db: Session = Depends(get_postgres_session)
):
    """Get document details and processing status"""
    try:
        document = db.query(Document).filter(Document.id == document_id).first()

        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        return {
            "id": str(document.id),
            "filename": document.original_filename,
            "filepath": document.filepath,
            "uploaded_at": document.uploaded_at.isoformat(),
            "processed_at": document.processed_at.isoformat() if document.processed_at else None,
            "is_processed": document.is_processed,
            "processing_status": document.processing_status,
            "total_chunks": document.total_chunks,
            "error_message": document.error_message
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==== Process Document Manually ====
@router.post("/documents/{document_id}/process")
async def process_document(
    document_id: str,
    db: Session = Depends(get_postgres_session)
):
    """Manually trigger document processing"""
    try:
        document = db.query(Document).filter(Document.id == document_id).first()

        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        if document.is_processed:
            return {
                "success": True,
                "message": "Document already processed",
                "document_id": str(document.id)
            }

        await process_document_pipeline(document, db)

        return {
            "success": True,
            "message": "Document processed successfully",
            "document_id": str(document.id),
            "total_chunks": document.total_chunks
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==== Delete Document ====
@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    db: Session = Depends(get_postgres_session)
):
    """Delete a document and its embeddings"""
    try:
        document = db.query(Document).filter(Document.id == document_id).first()

        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        # Delete embeddings from PGVector
        await vector_store.delete_document_embeddings(document.id)

        # Delete BM25 index
        await bm25_service.delete_index(str(document.id))

        # Delete files
        if os.path.exists(document.filepath):
            os.remove(document.filepath)
        if document.text_filepath and os.path.exists(document.text_filepath):
            os.remove(document.text_filepath)

        # Delete database record
        db.delete(document)
        db.commit()

        return {
            "success": True,
            "message": "Document deleted successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==== Graph Processing Endpoints ====
@router.post("/documents/{document_id}/process-graph")
async def process_document_graph(
    document_id: str,
    db: Session = Depends(get_postgres_session)
):
    """
    Process document with GraphRAG to extract knowledge graph

    Args:
        document_id: Document ID to process
        db: Database session

    Returns:
        Graph processing status and statistics
    """
    try:
        document = db.query(Document).filter(Document.id == document_id).first()

        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        if not document.is_processed or not document.text_filepath:
            raise HTTPException(
                status_code=400,
                detail="Document must be processed first (text extraction required)"
            )

        logger.info(f"Starting GraphRAG processing for document {document_id}")

        # Read extracted text
        with open(document.text_filepath, 'r', encoding='utf-8') as f:
            text_content = f.read()

        # Initialize GraphRAG pipeline for this request
        graphrag_pipeline = GraphRAGPipeline()

        # Process with GraphRAG (chunks text and extracts entities/relationships from each chunk)
        graph_result = await graphrag_pipeline.process_document(
            text_content=text_content,
            document_id=str(document_id)
        )

        # Import all graph documents into Neo4j (one per chunk)
        await neo4j_service.create_constraints()
        await neo4j_service.import_graph_documents(
            graph_documents=graph_result["graph_documents"],  # Now a list of all chunk graph docs
            document_id=str(document_id)
        )

        # Update document record with stats
        document.graph_processed = True
        document.graph_entities_count = graph_result["entities_count"]
        document.graph_relationships_count = graph_result["relationships_count"]
        document.graph_processing_time = graph_result["processing_time"]
        db.commit()

        logger.info(
            f"GraphRAG processing complete for {document_id}: "
            f"{graph_result['entities_count']} entities, "
            f"{graph_result['relationships_count']} relationships "
            f"from {graph_result['chunks_processed']} chunks"
        )

        return {
            "success": True,
            "document_id": str(document_id),
            "graph_processed": True,
            "entities_count": graph_result["entities_count"],
            "relationships_count": graph_result["relationships_count"],
            "chunks_processed": graph_result["chunks_processed"],
            "processing_time": graph_result["processing_time"],
            "message": f"Knowledge graph extracted from {graph_result['chunks_processed']} chunks and stored successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing graph for document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/stats/{document_id}")
async def get_graph_stats(
    document_id: str,
    db: Session = Depends(get_postgres_session)
):
    """Get graph statistics for a document"""
    try:
        document = db.query(Document).filter(Document.id == document_id).first()

        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        # Get stats from Neo4j
        graph_stats = await neo4j_service.get_document_stats(document_id)

        return {
            "document_id": document_id,
            "graph_processed": document.graph_processed,
            "entities_count": document.graph_entities_count,
            "relationships_count": document.graph_relationships_count,
            "processing_time": document.graph_processing_time,
            "neo4j_stats": graph_stats
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting graph stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/entities")
async def get_entities(
    document_id: Optional[str] = None,
    limit: int = 50
):
    """Get list of entities from the knowledge graph"""
    try:
        cypher = """
        MATCH (e:__Entity__)
        """

        if document_id:
            cypher += """
            WHERE (e)-[:BELONGS_TO]->(:__Document__ {id: $document_id})
            """

        cypher += """
        RETURN e.id as id,
               e.type as type,
               e.description as description
        LIMIT $limit
        """

        params = {"limit": limit}
        if document_id:
            params["document_id"] = document_id

        entities = await neo4j_service.query_graph(cypher, params)

        return {
            "entities": entities,
            "count": len(entities)
        }

    except Exception as e:
        logger.error(f"Error getting entities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/relationships")
async def get_relationships(
    document_id: Optional[str] = None,
    limit: int = 50
):
    """Get list of relationships from the knowledge graph"""
    try:
        cypher = """
        MATCH (e1:__Entity__)-[r]->(e2:__Entity__)
        WHERE NOT type(r) = 'BELONGS_TO'
        """

        if document_id:
            cypher += """
            AND (e1)-[:BELONGS_TO]->(:__Document__ {id: $document_id})
            """

        cypher += """
        RETURN e1.id as source,
               type(r) as type,
               e2.id as target
        LIMIT $limit
        """

        params = {"limit": limit}
        if document_id:
            params["document_id"] = document_id

        relationships = await neo4j_service.query_graph(cypher, params)

        return {
            "relationships": relationships,
            "count": len(relationships)
        }

    except Exception as e:
        logger.error(f"Error getting relationships: {e}")
        raise HTTPException(status_code=500, detail=str(e))

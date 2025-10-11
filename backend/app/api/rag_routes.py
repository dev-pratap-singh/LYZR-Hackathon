"""
RAG System API Routes
Handles document upload, query processing, and streaming responses
"""
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Literal
from uuid import UUID

from sqlalchemy.orm import Session

from app.database import get_postgres_session, get_pgvector_session
from app.models import Document, Embedding
from app.services.search_agent import SearchAgent
from app.services.vector_store import VectorStoreService
from app.services.document_processor import DocumentProcessingService
from app.services.bm25_search import BM25SearchService
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
        import os
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

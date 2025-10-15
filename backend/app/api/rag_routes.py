"""
RAG System API Routes
Handles document upload, query processing, and streaming responses
"""
import logging
import os
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Header
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
from app.services.elasticsearch_service import elasticsearch_service
from app.utils.document_helpers import process_document_pipeline
from app.config import settings

logger = logging.getLogger(__name__)

ProcessingMethod = Literal["pymupdf", "docling"]

router = APIRouter(prefix="/api/rag", tags=["RAG System"])

# Initialize services
vector_store = VectorStoreService()
doc_processor = DocumentProcessingService()
bm25_service = BM25SearchService()


# ==== Helper Functions ====
def get_openai_api_key(x_openai_api_key: Optional[str] = Header(None)) -> str:
    """
    Get OpenAI API key from header or fallback to settings.

    Args:
        x_openai_api_key: Optional API key from X-OpenAI-API-Key header

    Returns:
        API key to use (user-provided or system default)
    """
    # Use user-provided key if available, otherwise fallback to settings
    api_key = x_openai_api_key or settings.openai_api_key

    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="OpenAI API key not provided. Please set X-OpenAI-API-Key header or configure OPENAI_API_KEY in environment."
        )

    return api_key


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
    db: Session = Depends(get_postgres_session),
    openai_api_key: str = Depends(get_openai_api_key)
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
            await process_document_pipeline(document, db, method=method, openai_api_key=openai_api_key)
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
    db: Session = Depends(get_postgres_session),
    openai_api_key: str = Depends(get_openai_api_key)
):
    """
    Process query with streaming response

    - Generates embeddings for query
    - Performs vector search
    - Streams LLM response with reasoning

    Headers:
        X-OpenAI-API-Key: Optional user-provided OpenAI API key
    """
    try:
        # Create SearchAgent with user-provided API key
        search_agent = SearchAgent(openai_api_key=openai_api_key)

        # Ensure graph is processed for all documents (happens once on first query)
        await search_agent.ensure_graph_processed(db)

        # If document_id provided, verify it exists and process if needed
        document_id = None
        if request.document_id:
            document = db.query(Document).filter(Document.id == request.document_id).first()
            if not document:
                raise HTTPException(status_code=404, detail="Document not found")

            # Process document if not already processed
            if not document.is_processed:
                await process_document_pipeline(document, db, openai_api_key=openai_api_key)

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
    db: Session = Depends(get_postgres_session),
    openai_api_key: str = Depends(get_openai_api_key)
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

        await process_document_pipeline(document, db, openai_api_key=openai_api_key)

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

        # Delete graph data from Neo4j
        try:
            await neo4j_service.delete_document_graph(str(document.id))
            logger.info(f"Graph data deleted for document {document.id}")
        except Exception as e:
            logger.warning(f"Failed to delete graph data: {e}")

        # Delete from Elasticsearch
        try:
            await elasticsearch_service.delete_document(str(document.id))
            logger.info(f"Elasticsearch document deleted for {document.id}")
        except Exception as e:
            logger.warning(f"Failed to delete from Elasticsearch: {e}")

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
    db: Session = Depends(get_postgres_session),
    openai_api_key: str = Depends(get_openai_api_key)
):
    """
    Process document with GraphRAG to extract knowledge graph

    Args:
        document_id: Document ID to process
        db: Database session
        openai_api_key: User-provided OpenAI API key

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

        # Initialize GraphRAG pipeline for this request with user-provided API key
        graphrag_pipeline = GraphRAGPipeline(openai_api_key=openai_api_key)

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


@router.get("/graph/data")
async def get_graph_data(
    limit: int = 100
):
    """
    Get complete graph data (nodes and edges) for visualization
    Returns unified knowledge graph across all documents

    Args:
        limit: Maximum number of nodes to return

    Returns:
        Graph data in D3.js compatible format with nodes and edges
    """
    try:
        # Query to get entities (nodes) with their properties from ALL documents
        nodes_cypher = """
        MATCH (e:__Entity__)-[:BELONGS_TO]->(d:__Document__)
        RETURN e.id as id,
               e.id as label,
               labels(e) as types,
               e.description as description,
               properties(e) as properties
        LIMIT $limit
        """

        nodes_result = await neo4j_service.query_graph(
            nodes_cypher,
            {"limit": limit}
        )

        # Format nodes for D3.js
        nodes = []
        for node in nodes_result:
            # Get the entity type (first non-system label)
            entity_types = [t for t in node.get('types', []) if not t.startswith('__')]
            entity_type = entity_types[0] if entity_types else "Entity"

            nodes.append({
                "id": node['id'],
                "label": node['id'],
                "type": entity_type,
                "description": node.get('description', ''),
                "properties": node.get('properties', {})
            })

        # Query to get relationships (edges) across ALL documents
        edges_cypher = """
        MATCH (e1:__Entity__)-[:BELONGS_TO]->(d1:__Document__)
        MATCH (e1)-[r]->(e2:__Entity__)-[:BELONGS_TO]->(d2:__Document__)
        WHERE NOT type(r) = 'BELONGS_TO'
        RETURN e1.id as source,
               e2.id as target,
               type(r) as type,
               properties(r) as properties
        LIMIT $limit
        """

        edges_result = await neo4j_service.query_graph(
            edges_cypher,
            {"limit": limit}
        )

        # Format edges for D3.js
        edges = []
        edge_id = 0
        for edge in edges_result:
            edges.append({
                "id": f"edge_{edge_id}",
                "source": edge['source'],
                "target": edge['target'],
                "type": edge['type'],
                "label": edge['type'],
                "properties": edge.get('properties', {})
            })
            edge_id += 1

        # Get graph statistics across all documents
        total_entities = len(nodes)
        total_relationships = len(edges)

        return {
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "total_entities": total_entities,
                "total_relationships": total_relationships
            }
        }

    except Exception as e:
        logger.error(f"Error getting graph data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/search")
async def search_graph(
    query: str,
    limit: int = 20
):
    """
    Search for entities across all documents in the knowledge graph

    Args:
        query: Search query string
        limit: Maximum number of results

    Returns:
        List of matching entities
    """
    try:
        cypher = """
        MATCH (e:__Entity__)-[:BELONGS_TO]->(d:__Document__)
        WHERE toLower(e.id) CONTAINS toLower($query)
        OR toLower(e.description) CONTAINS toLower($query)
        RETURN e.id as id,
               e.id as label,
               labels(e) as types,
               e.description as description,
               properties(e) as properties
        LIMIT $limit
        """

        params = {"query": query, "limit": limit}

        results = await neo4j_service.query_graph(cypher, params)

        # Format results
        entities = []
        for result in results:
            entity_types = [t for t in result.get('types', []) if not t.startswith('__')]
            entity_type = entity_types[0] if entity_types else "Entity"

            entities.append({
                "id": result['id'],
                "label": result['id'],
                "type": entity_type,
                "description": result.get('description', ''),
                "properties": result.get('properties', {})
            })

        return {
            "entities": entities,
            "count": len(entities),
            "query": query
        }

    except Exception as e:
        logger.error(f"Error searching graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/graph/clear")
async def clear_all_graph_data(
    db: Session = Depends(get_postgres_session)
):
    """
    Clear all graph data from Neo4j - useful for starting fresh

    Returns:
        Success status and message
    """
    try:
        logger.info("API request to clear all graph data")

        # Clear all data from Neo4j
        await neo4j_service.clear_all_graph_data()

        # Reset graph_processed flag for all documents
        documents = db.query(Document).filter(Document.graph_processed == True).all()
        for doc in documents:
            doc.graph_processed = False
            doc.graph_entities_count = 0
            doc.graph_relationships_count = 0
        db.commit()

        logger.info(f"All graph data cleared. Reset {len(documents)} document(s) graph_processed flag")

        return {
            "success": True,
            "message": f"All graph data cleared successfully. {len(documents)} document(s) reset.",
            "documents_reset": len(documents)
        }

    except Exception as e:
        logger.error(f"Error clearing graph data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/graph/cleanup-orphans")
async def cleanup_orphaned_relationships():
    """
    Remove orphaned relationships (edges that reference non-existent nodes)

    This fixes data integrity issues where relationships point to deleted nodes.

    Returns:
        Number of orphaned relationships removed
    """
    try:
        logger.info("API request to cleanup orphaned relationships")

        # Query to find and delete orphaned relationships
        # Relationships where either source or target node doesn't exist
        cleanup_cypher = """
        MATCH ()-[r]->()
        WHERE NOT EXISTS {
            MATCH (start)-[r]->(end)
            WHERE (start:__Entity__ OR start:__Document__)
              AND (end:__Entity__ OR end:__Document__)
        }
        DELETE r
        RETURN count(r) as deleted_count
        """

        result = await neo4j_service.query_graph(cleanup_cypher, {})
        deleted_count = result[0]["deleted_count"] if result else 0

        logger.info(f"Cleanup complete: {deleted_count} orphaned relationships removed")

        return {
            "success": True,
            "message": f"Removed {deleted_count} orphaned relationship(s)",
            "deleted_count": deleted_count
        }

    except Exception as e:
        logger.error(f"Error cleaning up orphaned relationships: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==== Elasticsearch / Filter Search Endpoints ====
class FilterSearchRequest(BaseModel):
    query: str = ""
    filters: Optional[dict] = None
    size: int = 10
    from_: int = 0


@router.post("/search/filter")
async def filter_search(request: FilterSearchRequest):
    """
    Search documents using Elasticsearch with metadata filters

    Args:
        request: Search query and filter parameters

    Returns:
        Filtered search results
    """
    try:
        results = await elasticsearch_service.search(
            query=request.query,
            filters=request.filters,
            size=request.size,
            from_=request.from_
        )

        return {
            "success": True,
            "results": results
        }

    except Exception as e:
        logger.error(f"Error in filter search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search/aggregations/{field}")
async def get_search_aggregations(field: str):
    """
    Get aggregations (facets) for a field

    Args:
        field: Field to aggregate (author, categories, tags, etc.)

    Returns:
        Aggregation results
    """
    try:
        aggregations = await elasticsearch_service.get_aggregations(field)

        return {
            "success": True,
            "aggregations": aggregations
        }

    except Exception as e:
        logger.error(f"Error getting aggregations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/filters")
async def get_available_filters():
    """
    Get available filter options (authors, categories, tags)

    Returns:
        Available filter values for all fields
    """
    try:
        # Get aggregations for multiple fields
        authors = await elasticsearch_service.get_aggregations("author")
        categories = await elasticsearch_service.get_aggregations("categories")
        tags = await elasticsearch_service.get_aggregations("tags")
        doc_types = await elasticsearch_service.get_aggregations("document_type")

        return {
            "success": True,
            "filters": {
                "authors": authors.get("values", []),
                "categories": categories.get("values", []),
                "tags": tags.get("values", []),
                "document_types": doc_types.get("values", [])
            }
        }

    except Exception as e:
        logger.error(f"Error getting available filters: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/{document_id}/reindex")
async def reindex_document(
    document_id: str,
    db: Session = Depends(get_postgres_session)
):
    """
    Reindex a specific document in Elasticsearch

    Args:
        document_id: Document ID to reindex
        db: Database session

    Returns:
        Reindexing status
    """
    try:
        document = db.query(Document).filter(Document.id == document_id).first()

        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        if not document.is_processed or not document.text_filepath:
            raise HTTPException(
                status_code=400,
                detail="Document must be processed first"
            )

        # Create index if needed
        elasticsearch_service.create_index()

        # Read text content
        with open(document.text_filepath, 'r', encoding='utf-8') as f:
            text_content = f.read()

        # Reindex
        success = await elasticsearch_service.index_document(
            document_id=str(document.id),
            content=text_content,
            filename=document.original_filename,
            author=document.author,
            document_type=document.document_type or "pdf",
            categories=document.categories if isinstance(document.categories, list) else [],
            tags=document.tags if isinstance(document.tags, list) else [],
            uploaded_at=document.uploaded_at,
            processed_at=document.processed_at,
            file_size=document.file_size,
            chunk_count=document.total_chunks,
            user_id=document.user_id,
            metadata=document.doc_metadata
        )

        if success:
            from datetime import datetime
            document.elasticsearch_indexed = True
            document.elasticsearch_index_time = datetime.now()
            db.commit()

            return {
                "success": True,
                "message": f"Document {document_id} reindexed successfully"
            }
        else:
            return {
                "success": False,
                "message": "Reindexing failed"
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reindexing document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/elasticsearch/create-index")
async def create_elasticsearch_index():
    """
    Create Elasticsearch index with proper mappings

    Returns:
        Index creation status
    """
    try:
        success = elasticsearch_service.create_index()

        if success:
            return {
                "success": True,
                "message": "Elasticsearch index created successfully"
            }
        else:
            return {
                "success": False,
                "message": "Failed to create index"
            }

    except Exception as e:
        logger.error(f"Error creating Elasticsearch index: {e}")
        raise HTTPException(status_code=500, detail=str(e))

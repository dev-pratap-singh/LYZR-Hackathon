import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.rag_routes import router as rag_router
from app.api.memory_routes import router as memory_router
from app.config import settings
from app.database import init_databases
from app.services.elasticsearch_service import elasticsearch_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("🚀 Starting RAG System...")

    try:
        # Initialize databases
        logger.info("Initializing databases...")
        init_databases()
        logger.info("✅ Databases initialized successfully")

        # Initialize Elasticsearch index (if enabled)
        if settings.enable_filter_search:
            try:
                logger.info("Initializing Elasticsearch index...")
                elasticsearch_service.create_index()
                logger.info("✅ Elasticsearch index ready")
            except Exception as es_error:
                logger.warning(f"⚠️ Elasticsearch initialization failed: {es_error}")
                logger.warning("Filter search will not be available")

    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        raise

    logger.info("✅ RAG System ready!")

    yield

    # Shutdown
    logger.info("Shutting down RAG System...")

    # Close Elasticsearch connections
    if settings.enable_filter_search:
        try:
            await elasticsearch_service.close()
        except Exception as e:
            logger.error(f"Error closing Elasticsearch: {e}")


app = FastAPI(
    title="RAG System API",
    description="Advanced RAG system with Vector, Graph, and Filter search",
    version="2.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(rag_router)
app.include_router(memory_router)


@app.get("/")
async def root():
    return {
        "message": "RAG System API",
        "status": "running",
        "version": "2.0.0",
        "features": [
            "PDF document upload",
            "Document processing with PyMuPDF and Docling",
            "Vector search with PGVector and hybrid retrieval (Vector + BM25)",
            "Graph search with Neo4j knowledge graphs",
            "Filter search with Elasticsearch metadata filtering",
            "Multi-tool search agent with automatic tool selection",
            "Streaming responses with Chain of Thought reasoning",
            "OpenAI embeddings and LLM"
        ]
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "storage_path": settings.storage_path,
        "openai_configured": settings.openai_api_key is not None
    }

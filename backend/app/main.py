import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.rag_routes import router as rag_router
from app.config import settings
from app.database import init_databases

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

    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        raise

    logger.info("✅ RAG System ready!")

    yield

    # Shutdown
    logger.info("Shutting down RAG System...")


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


@app.get("/")
async def root():
    return {
        "message": "RAG System API",
        "status": "running",
        "version": "2.0.0",
        "features": [
            "PDF document upload",
            "Document processing with Docling",
            "Vector search with PGVector",
            "Streaming responses with CoT reasoning",
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

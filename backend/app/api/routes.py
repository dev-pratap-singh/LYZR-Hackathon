from fastapi import APIRouter, HTTPException
from typing import Dict, Any

router = APIRouter(prefix="/api", tags=["api"])


@router.get("/status")
async def get_status() -> Dict[str, Any]:
    """Get system status"""
    return {
        "backend": "operational",
        "database": "connected",
        "storage": "accessible"
    }


@router.get("/search/test")
async def test_search() -> Dict[str, Any]:
    """Test search endpoint (placeholder)"""
    return {
        "message": "Search endpoint placeholder",
        "tools": ["vector_search", "graph_search", "filter_search"]
    }

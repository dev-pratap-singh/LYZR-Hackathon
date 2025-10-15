"""
Memory Management API Routes
Provides endpoints for accessing memory state, token usage, and working memory
"""
import logging
from fastapi import APIRouter, HTTPException
from typing import Optional
from pydantic import BaseModel

from app.config import settings
from app.services.memory import MemoryManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/memory", tags=["Memory Management"])

# Global memory manager instance (shared across requests)
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Get or create the global memory manager instance"""
    global _memory_manager

    if not settings.memory_enabled:
        raise HTTPException(status_code=503, detail="Memory management is disabled")

    if _memory_manager is None:
        try:
            _memory_manager = MemoryManager(
                db_host=settings.memory_db_host,
                db_port=settings.memory_db_port,
                db_name=settings.memory_db_name,
                db_user=settings.memory_db_user,
                db_password=settings.memory_db_password,
                model_name=settings.memory_model,
                openai_api_key=settings.openai_api_key,
                session_id=settings.memory_session_id
            )
            logger.info("Memory Manager initialized for API")
        except Exception as e:
            logger.error(f"Failed to initialize Memory Manager: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize memory manager: {str(e)}")

    return _memory_manager


@router.get("/state")
async def get_memory_state():
    """
    Get current memory state including context utilization and token usage

    Returns:
        Memory state with statistics
    """
    try:
        memory_manager = get_memory_manager()
        state = memory_manager.export_memory_state_json()

        return {
            "success": True,
            "data": state
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting memory state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/token-usage")
async def get_token_usage():
    """
    Get token usage statistics

    Returns:
        Token usage stats including input/output tokens, cost, and percentage used
    """
    try:
        memory_manager = get_memory_manager()
        state = memory_manager.get_memory_state()

        token_stats = state.get('token_usage_stats', {})
        context_length = state.get('total_context_length', 0)

        total_tokens = int(token_stats.get('input_tokens', 0)) + int(token_stats.get('output_tokens', 0))
        percentage_used = (total_tokens / context_length * 100) if context_length > 0 else 0

        return {
            "success": True,
            "data": {
                "input_tokens": int(token_stats.get('input_tokens', 0)),
                "output_tokens": int(token_stats.get('output_tokens', 0)),
                "total_tokens": total_tokens,
                "total_cost": float(token_stats.get('total_cost', 0)),
                "context_limit": context_length,
                "percentage_used": round(percentage_used, 2),
                "tokens_remaining": context_length - total_tokens
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting token usage: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/working-memory")
async def get_working_memory(limit: int = 10):
    """
    Get working memory items

    Args:
        limit: Maximum number of items to return (default: 10)

    Returns:
        List of recent memory items
    """
    try:
        memory_manager = get_memory_manager()
        items = memory_manager.get_working_memory(limit=limit)

        return {
            "success": True,
            "data": {
                "items": items,
                "count": len(items)
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting working memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/{session_id}/state")
async def get_session_memory_state(session_id: str):
    """
    Get memory state for a specific session

    Args:
        session_id: Session ID to query

    Returns:
        Memory state for the specified session
    """
    try:
        # Create a temporary memory manager for this session
        temp_manager = MemoryManager(
            db_host=settings.memory_db_host,
            db_port=settings.memory_db_port,
            db_name=settings.memory_db_name,
            db_user=settings.memory_db_user,
            db_password=settings.memory_db_password,
            model_name=settings.memory_model,
            openai_api_key=settings.openai_api_key,
            session_id=session_id
        )

        state = temp_manager.export_memory_state_json()
        temp_manager.close()

        return {
            "success": True,
            "data": state
        }
    except Exception as e:
        logger.error(f"Error getting session memory state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/clear")
async def clear_memory():
    """
    Clear all memory items and reset memory state for the current session

    Returns:
        Success status and statistics about what was cleared
    """
    try:
        memory_manager = get_memory_manager()
        result = memory_manager.clear_memory()

        if result['success']:
            return {
                "success": True,
                "message": result['message'],
                "items_cleared": result['items_cleared'],
                "session_id": result['session_id']
            }
        else:
            raise HTTPException(status_code=500, detail=result.get('message', 'Failed to clear memory'))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def memory_health_check():
    """
    Check if memory management system is healthy

    Returns:
        Health status
    """
    try:
        if not settings.memory_enabled:
            return {
                "success": True,
                "status": "disabled",
                "message": "Memory management is disabled"
            }

        memory_manager = get_memory_manager()
        state = memory_manager.get_memory_state()

        return {
            "success": True,
            "status": "healthy",
            "message": "Memory management system is operational",
            "session_id": memory_manager.session_id,
            "model": memory_manager.model_name
        }
    except Exception as e:
        logger.error(f"Memory health check failed: {e}")
        return {
            "success": False,
            "status": "unhealthy",
            "message": str(e)
        }

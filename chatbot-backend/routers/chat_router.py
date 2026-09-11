from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query
from models.models import (
    ChatHistory,
    ChatMessage,
    InitSessionRequest,
    PaginatedChatHistory,
)
from services.service_factory import ServiceFactory
from utils.logger import logger

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/", response_model=Dict[str, Any])
async def send_message(message: ChatMessage):
    """Send message to AI agents"""
    try:
        logger.log_api_request("POST", "/chat", message.session_id)

        chat_service = ServiceFactory.get_chat_service()
        result = await chat_service.send_message(message)

        logger.log_api_response("POST", "/chat", 200, message.session_id)
        return result

    except HTTPException:
        # Preserve HTTPException (status code + message) from service layer
        raise
    except Exception as e:
        logger.error(f"Error in send_message endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/init_session", response_model=Dict[str, Any])
async def init_session(body: InitSessionRequest):
    """Initialize session content (greeting rewrite or passthrough by request_type)."""
    try:
        logger.log_api_request("POST", "/chat/init_session", body.session_id)

        init_session_service = ServiceFactory.get_init_session_service()
        result = await init_session_service.init_session(body)

        logger.log_api_response("POST", "/chat/init_session", 200, body.session_id)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in init_session endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{session_id}", response_model=ChatHistory)
async def get_chat_history(session_id: str):
    """Get chat history for a session"""
    try:
        logger.log_api_request("GET", f"/chat/history/{session_id}", session_id)

        chat_service = ServiceFactory.get_chat_service()
        result = await chat_service.get_chat_history(session_id)

        logger.log_api_response("GET", f"/chat/history/{session_id}", 200, session_id)
        return result

    except Exception as e:
        logger.error(f"Error in get_chat_history endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/history/{session_id}")
async def clear_chat_history(session_id: str):
    """Clear chat history for a session"""
    try:
        logger.log_api_request("DELETE", f"/chat/history/{session_id}", session_id)

        chat_service = ServiceFactory.get_chat_service()
        result = await chat_service.clear_chat_history(session_id)

        logger.log_api_response(
            "DELETE", f"/chat/history/{session_id}", 200, session_id
        )
        return result

    except Exception as e:
        logger.error(f"Error in clear_chat_history endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/user/{user_id}", response_model=PaginatedChatHistory)
async def get_chat_history_by_user_id(
    user_id: str,
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    limit: int = Query(20, ge=1, le=100, description="Number of messages per page"),
    sort_order: str = Query(
        "desc",
        regex="^(asc|desc)$",
        description="Sort order: 'asc' (oldest first) or 'desc' (newest first)",
    ),
):
    """
    Get chat history for a user_id with pagination and sorting by timestamp

    Args:
        user_id: The user ID to get history for
        page: Page number (1-indexed, default: 1)
        limit: Number of messages per page (default: 20, max: 100)
        sort_order: Sort order - "asc" (oldest first) or "desc" (newest first, default)

    Returns:
        PaginatedChatHistory with messages, total, page, limit, total_pages
    """
    try:
        logger.log_api_request(
            "GET", f"/chat/history/user/{user_id}", f"user_{user_id}"
        )

        chat_service = ServiceFactory.get_chat_service()
        result = await chat_service.get_chat_history_by_user_id(
            user_id=user_id, page=page, limit=limit, sort_order=sort_order
        )

        logger.log_api_response(
            "GET", f"/chat/history/user/{user_id}", 200, f"user_{user_id}"
        )
        return result

    except Exception as e:
        logger.error(f"Error in get_chat_history_by_user_id endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

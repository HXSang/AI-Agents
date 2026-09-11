from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from models.models import ChatHistory, ChatMessage, ChatResponse


class ChatService(ABC):
    """Abstract interface for chat service operations"""

    @abstractmethod
    async def send_message(self, message: ChatMessage) -> Dict[str, Any]:
        """Send message to AI agents"""
        pass

    @abstractmethod
    async def get_chat_history(self, session_id: str) -> ChatHistory:
        """Get chat history for a session"""
        pass

    @abstractmethod
    async def clear_chat_history(self, session_id: str) -> Dict[str, str]:
        """Clear chat history for a session"""
        pass

    @abstractmethod
    async def get_chat_history_by_user_id(
        self,
        user_id: str,
        page: int = 1,
        limit: int = 20,
        sort_order: str = "desc",
    ) -> Dict[str, Any]:
        """
        Get chat history for a user_id with pagination and sorting

        Args:
            user_id: The user ID to get history for
            page: Page number (1-indexed)
            limit: Number of messages per page
            sort_order: Sort order - "asc" (oldest first) or "desc" (newest first)

        Returns:
            Dict with messages, total, page, limit, total_pages
        """
        pass

    @abstractmethod
    async def handle_ai_response(self, response_data: Dict[str, Any]) -> None:
        """Handle response from AI agents"""
        pass

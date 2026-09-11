from abc import ABC, abstractmethod
from typing import Any, Dict

from fastapi import WebSocket


class WebSocketService(ABC):
    """Abstract interface for WebSocket service operations"""

    @abstractmethod
    async def connect(self, websocket: WebSocket, session_id: str) -> None:
        """Connect a WebSocket client"""
        pass

    @abstractmethod
    def disconnect(self, session_id: str) -> None:
        """Disconnect a WebSocket client"""
        pass

    @abstractmethod
    async def send_message(self, session_id: str, message: Dict[str, Any]) -> None:
        """Send message to a WebSocket client"""
        pass

    @abstractmethod
    def get_active_connections(self) -> Dict[str, WebSocket]:
        """Get all active WebSocket connections"""
        pass

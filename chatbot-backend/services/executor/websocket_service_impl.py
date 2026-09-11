from typing import Any, Dict

from fastapi import WebSocket
from services.websocket_service import WebSocketService
from utils.logger import logger


class WebSocketServiceImpl(WebSocketService):
    """Implementation of WebSocket service"""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_id: str) -> None:
        """Connect a WebSocket client"""
        try:
            await websocket.accept()
            self.active_connections[session_id] = websocket

            logger.log_websocket_connection(session_id, "connected")

        except Exception as e:
            logger.error(f"Error connecting WebSocket: {str(e)}")
            raise

    def disconnect(self, session_id: str) -> None:
        """Disconnect a WebSocket client"""
        try:
            if session_id in self.active_connections:
                del self.active_connections[session_id]

                logger.log_websocket_connection(session_id, "disconnected")

        except Exception as e:
            logger.error(f"Error disconnecting WebSocket: {str(e)}")

    async def send_message(self, session_id: str, message: Dict[str, Any]) -> None:
        """Send message to a WebSocket client"""
        try:
            if session_id in self.active_connections:
                websocket = self.active_connections[session_id]
                await websocket.send_json(message)

                logger.log_websocket_connection(
                    session_id,
                    "message_sent",
                    message_type=message.get("type", "unknown"),
                )
            else:
                logger.warning(f"No active connection found for session {session_id}")

        except Exception as e:
            logger.error(f"Error sending WebSocket message: {str(e)}")
            self.disconnect(session_id)

    def get_active_connections(self) -> Dict[str, WebSocket]:
        """Get all active WebSocket connections"""
        return self.active_connections.copy()

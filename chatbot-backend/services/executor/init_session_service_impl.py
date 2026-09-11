import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from clients.rabbitmq_client import RabbitMQClient
from clients.redis_client import RedisClient
from fastapi import HTTPException
from models.models import InitSessionRequest, RabbitMQMessage
from services.executor.chat_service_impl import ChatServiceImpl
from services.init_session_service import InitSessionService
from utils.logger import logger


class InitSessionServiceImpl(InitSessionService):
    """Publish init_session jobs to the same queue as chat."""

    def __init__(self):
        self.rabbitmq_client: Optional[RabbitMQClient] = None
        self.redis_client = RedisClient()
        self._chat_service: Optional[ChatServiceImpl] = None

    def set_rabbitmq_client(self, rabbitmq_client: RabbitMQClient) -> None:
        self.rabbitmq_client = rabbitmq_client

    def set_chat_service(self, chat_service: ChatServiceImpl) -> None:
        self._chat_service = chat_service

    async def init_session(self, request: InitSessionRequest) -> Dict[str, Any]:
        try:
            user_id = request.user_id
            if not user_id:
                raise HTTPException(
                    status_code=400, detail="User ID is missing when sending message"
                )

            chat_service = self._chat_service
            if not chat_service:
                raise HTTPException(
                    status_code=500,
                    detail="Chat service not configured for init_session",
                )

            user_exists = await chat_service._ensure_general_cache(user_id)
            if not user_exists:
                raise HTTPException(
                    status_code=400,
                    detail=f"User not found when ensuring general cache: {user_id}",
                )

            session_id = request.session_id or str(uuid.uuid4())
            message_id = str(uuid.uuid4())
            flow = "init_session"
            message = ""
            if request.request_type == "greeting":
                message = "Hi"

            rabbitmq_message = RabbitMQMessage(
                message_id=message_id,
                request_id=request.request_id,
                user_id=user_id,
                session_id=session_id,
                app_id=request.app_id,
                message=message,
                flow=flow,
                timezone=request.timezone,
                request_type=request.request_type,
                request_content=request.request_content,
                language=request.language,
                is_auto_user_message=True,
            )

            await self.redis_client.store_pending_message(
                message_id,
                {
                    "session_id": session_id,
                    "user_id": user_id,
                    "request_id": request.request_id,
                    "app_id": request.app_id,
                    "message": message,
                    "is_auto_user_message": True,
                    "request_type": request.request_type,
                    "request_content": request.request_content,
                    "language": request.language,
                    "flow": flow,
                    "timestamp": datetime.now().isoformat(),
                },
            )

            if self.rabbitmq_client:
                await self.rabbitmq_client.send_message(rabbitmq_message)
            else:
                logger.warning(
                    "RabbitMQ client not available, init_session not sent to AI agents"
                )

            return {
                "message_id": message_id,
                "session_id": session_id,
                "status": "sent",
                "message": "Init session request sent to AI agents",
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in init_session: {str(e)}")
            raise

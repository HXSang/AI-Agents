import asyncio
import json
import uuid
from datetime import datetime
from typing import Any, Dict

from clients.rabbitmq_client import RabbitMQClient
from clients.redis_client import RedisClient
from fastapi import HTTPException
from models.models import (
    ChatHistory,
    ChatMessage,
    ChatResponse,
    ConversationPayload,
    RabbitMQMessage,
)
from services.chat_service import ChatService
from utils.logger import logger


class ChatServiceImpl(ChatService):
    """Implementation of chat service"""

    def __init__(self):
        self.rabbitmq_client = None
        self.redis_client = RedisClient()
        self.external_api_service = None
        self.websocket_service = None
        self._cache_ttl = 86400 * 30  # 30 days

    def set_rabbitmq_client(self, rabbitmq_client: RabbitMQClient):
        """Set RabbitMQ client instance"""
        self.rabbitmq_client = rabbitmq_client

    def set_external_api_service(self, external_api_service):
        """Set external API service instance"""
        self.external_api_service = external_api_service

    def set_websocket_service(self, websocket_service):
        """Set WebSocket service instance"""
        self.websocket_service = websocket_service

    async def send_message(self, message: ChatMessage) -> Dict[str, Any]:
        """Send message to AI agents"""
        try:
            logger.log_chat_message(
                f"Sending message to AI agents: {message.message}",
                session_id=message.session_id,
                user_message=message.message,
            )

            # Generate session_id if not provided
            session_id = message.session_id or str(uuid.uuid4())

            # Generate message_id
            message_id = str(uuid.uuid4())

            # Get user_id and flow
            # Always use explicit user_id (no more 'anonymous' fallback)
            user_id = message.user_id
            flow = message.flow or "general"

            # Always check general cache / user existence before sending.
            # If user_id is missing or user cannot be loaded, return error "User not found".
            if not user_id:
                logger.warning("User ID is missing when sending message")
                raise HTTPException(
                    status_code=400, detail="User ID is missing when sending message"
                )

            user_exists = await self._ensure_general_cache(user_id)
            if not user_exists:
                logger.warning(f"User not found when ensuring general cache: {user_id}")
                raise HTTPException(
                    status_code=400,
                    detail=f"User not found when ensuring general cache: {user_id}",
                )

            rabbitmq_message = RabbitMQMessage(
                message_id=message_id,
                request_id=message.request_id,
                user_id=user_id,
                session_id=session_id,
                app_id=message.app_id,
                message=message.message,
                flow=flow,
                timezone=message.timezone,
                quick_reply=message.quick_reply,  # Pass quick_reply from ChatMessage
                persist_history=message.persist_history,
            )

            # Store pending message
            await self.redis_client.store_pending_message(
                message_id,
                {
                    "session_id": session_id,
                    "user_id": message.user_id,
                    "request_id": message.request_id,
                    "app_id": message.app_id,
                    "message": message.message,  # Store user message for conversation saving
                    "flow": message.flow,
                    "persist_history": message.persist_history,
                    "timestamp": datetime.now().isoformat(),
                },
            )

            # Send to AI agents
            if self.rabbitmq_client:
                await self.rabbitmq_client.send_message(rabbitmq_message)
            else:
                logger.warning(
                    "RabbitMQ client not available, message not sent to AI agents"
                )

            logger.log_api_response("POST", "/chat", 200, session_id)

            return {
                "message_id": message_id,
                "session_id": session_id,
                "status": "sent",
                "message": "Message sent to AI agents",
            }

        except Exception as e:
            logger.error(f"Error sending message: {str(e)}")
            raise

    async def get_chat_history(self, session_id: str) -> ChatHistory:
        """Get chat history for a session"""
        try:
            # Get user_id from session if available, otherwise None
            # This allows checking long-term memory if needed
            messages = await self.redis_client.get_chat_history(
                session_id, user_id=None
            )

            logger.log_api_request("GET", f"/chat/history/{session_id}", session_id)

            return ChatHistory(
                session_id=session_id, messages=messages, flow="general"  # Default flow
            )

        except Exception as e:
            logger.error(f"Error getting chat history: {str(e)}")
            raise

    async def clear_chat_history(self, session_id: str) -> Dict[str, str]:
        """Clear chat history for a session"""
        try:
            await self.redis_client.clear_chat_session(session_id)

            logger.log_api_request("DELETE", f"/chat/history/{session_id}", session_id)

            return {"message": f"Chat history cleared for session {session_id}"}

        except Exception as e:
            logger.error(f"Error clearing chat history: {str(e)}")
            raise

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
        try:
            # Use long_term_key method from RedisClient
            result = await self.redis_client.get_chat_history_by_user_id(
                user_id=user_id, page=page, limit=limit, sort_order=sort_order
            )

            logger.log_api_request(
                "GET", f"/chat/history/user/{user_id}", f"user_{user_id}"
            )

            return result

        except Exception as e:
            logger.error(f"Error getting chat history by user_id: {str(e)}")
            raise

    async def handle_ai_response(self, response_data: Dict[str, Any]) -> None:
        """Handle response from AI agents"""
        try:
            message_id = response_data.get("message_id")
            session_id = response_data.get("session_id")

            if not message_id or not session_id:
                logger.error("Invalid response data: missing message_id or session_id")
                return

            # Get pending message data
            pending_data = await self.redis_client.get_pending_message(message_id)
            if not pending_data:
                logger.error(f"No pending data found for message {message_id}")
                return

            # Extract set_goal from extracted_info if present
            # Handle case where extracted_info might be None
            extracted_info = response_data.get("extracted_info") or {}
            set_goal = (
                extracted_info.get("set_goal", {})
                if isinstance(extracted_info, dict)
                else {}
            )

            # Create chat response
            chat_response = ChatResponse(
                response=response_data.get("response", ""),
                session_id=session_id,
                user_id=response_data.get("user_id"),
                flow=response_data.get("flow", "general"),
                message_id=message_id,
                connect_action=response_data.get("connect_action", {}),
                flow_status=response_data.get("flow_status", "in_progress"),
                quick_reply=response_data.get("quick_reply", []),
                set_goal=set_goal,
                action_completed=response_data.get("action_completed", ""),
            )

            user_id = response_data.get("user_id")
            if user_id and pending_data.get("persist_history", True):
                # Get user message from pending_data
                user_message = pending_data.get("message", "")
                await self.redis_client.save_chat_message(
                    session_id=session_id,
                    user_id=user_id,
                    message=user_message,
                    response=chat_response.response,
                    language=response_data.get("language", ""),
                    quick_reply=response_data.get("quick_reply"),
                    flow=response_data.get("flow", "general"),
                )

            # Check if onboarding is completed and handle external API integration
            flow_status = response_data.get("flow_status", "in_progress")
            flow = response_data.get("flow", "general")

            if flow_status == "completed" and flow == "onboarding":
                await self._handle_completed_onboarding(response_data)

            # Send response via WebSocket
            if self.websocket_service:
                await self.websocket_service.send_message(
                    session_id,
                    {"type": "response", "data": chat_response.model_dump(mode="json")},
                )
            else:
                logger.warning(
                    "WebSocket service not available, response not sent via WebSocket"
                )

            user_id = response_data.get("user_id")
            if (
                user_id
                and self.external_api_service
                and pending_data.get("persist_history", True)
            ):
                asyncio.create_task(
                    self._save_conversation_async(
                        user_id=user_id,
                        pending_data=pending_data,
                        response_data=response_data,
                    )
                )

            # Clean up pending message
            await self.redis_client.remove_pending_message(message_id)

            logger.info(f"Processed AI response for session {session_id}")

        except Exception as e:
            logger.error(f"Error handling AI response: {str(e)}")
            raise

    async def _handle_completed_onboarding(self, response_data: Dict[str, Any]) -> None:
        """Handle completed onboarding and save to external API"""
        try:
            user_id = response_data.get("user_id")
            # Handle case where extracted_info might be None
            extracted_info = response_data.get("extracted_info") or {}
            user_data = (
                extracted_info.get("user_data", {})
                if isinstance(extracted_info, dict)
                else {}
            )

            if not user_id:
                logger.warning("No user_id found in completed onboarding response")
                return

            logger.info(
                f"🎉 Onboarding completed for user {user_id} with {response_data}"
            )
            logger.info(f"📊 User data keys: {list(user_data.keys())}")

            # Check if external API service is available
            if not self.external_api_service:
                logger.warning("External API service not available, skipping data save")
                return

            # Save to external API
            logger.info(f"💾 Saving onboarding data to external API for user {user_id}")
            success = await self.external_api_service.save_onboarding_data(user_data)

            if success:
                logger.info(
                    f"✅ Successfully saved onboarding data to external API for user {user_id}"
                )
            else:
                logger.error(
                    f"❌ Failed to save onboarding data to external API for user {user_id}"
                )

        except Exception as e:
            logger.error(f"❌ Error handling completed onboarding: {str(e)}")
            # Don't raise exception to avoid breaking the main flow

    async def _save_conversation_async(
        self,
        user_id: str,
        pending_data: Dict[str, Any],
        response_data: Dict[str, Any],
    ) -> None:
        """Save conversation to database asynchronously (fire-and-forget)

        Args:
            user_id: User ID
            pending_data: Pending message data from Redis (contains userMessage)
            response_data: AI response data
            chat_response: ChatResponse object
        """
        try:
            # Extract user message from pending_data
            user_message = pending_data.get("message", "")

            # Extract bot response
            bot_response = response_data.get("response", "")

            # Extract language
            language = response_data.get("language", "")

            # Get flow
            flow = response_data.get("flow", "general")

            # Get profile_id from cache or API
            profile_id = await self.redis_client.get_profile_id_from_cache(user_id)

            if not profile_id and self.external_api_service:
                # If not in cache, fetch from API
                logger.info(
                    f"🔍 profile_id not in cache for user {user_id}, fetching from API..."
                )
                profile_id = await self.external_api_service.get_profile_id(user_id)

            # Fallback to user_id if profile_id still not found
            if not profile_id:
                logger.warning(
                    f"⚠️ profile_id not found for user {user_id}, using user_id as fallback"
                )
                profile_id = user_id

            # Create payload using model
            # Format timestamp as ISO 8601 without milliseconds (API requirement)
            now = datetime.utcnow()
            timestamp = (
                now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            )  # Remove last 3 digits (microseconds) and add Z

            payload = ConversationPayload(
                profileId=profile_id,
                timestamp=timestamp,
                userMessage=user_message,
                botResponse=bot_response,
                flow=flow,
                language=language,
            )

            # Save to database via external API
            success = await self.external_api_service.save_conversation(
                user_id=user_id,
                payload=payload.model_dump(),
            )

            if success:
                logger.info(f"✅ Successfully saved conversation for user {user_id}")
            else:
                logger.warning(f"⚠️ Failed to save conversation for user {user_id}")

        except Exception as e:
            logger.error(f"❌ Error saving conversation asynchronously: {str(e)}")
            # Don't raise exception - this is fire-and-forget

    # -------------------- General cache methods --------------------
    async def _ensure_general_cache(self, user_id: str) -> bool:
        """
        Ensure general cache exists for user.
        Returns True if user data exists (and cache is present/created),
        False if user does not exist or cannot be loaded.

        Check order:
        1. General cache
        2. Onboarding cache (onboarding:{user_id})
        3. External API (if available)
        """
        try:
            # Check if general cache exists
            general_data = await self.redis_client.get_general_user_data(user_id)

            if general_data:
                logger.info(f"✅ General cache already exists for user {user_id}")
                return True

            # General cache not found, check onboarding cache
            logger.info(
                f"🔍 General cache not found for user {user_id}, checking onboarding cache..."
            )

            onboarding_key = f"onboarding:{user_id}"
            onboarding_data = await self.redis_client.get_data(onboarding_key)

            if onboarding_data:
                logger.info(
                    f"✅ Onboarding cache found for user {user_id}, user exists"
                )
                return True

            # Neither general cache nor onboarding cache found, fetch from DB via external API
            logger.info(
                f"🔍 Neither general nor onboarding cache found for user {user_id}, fetching from DB..."
            )

            if not self.external_api_service:
                logger.warning(
                    "⚠️ External API service not available, cannot fetch user data"
                )
                return False

            # Fetch user data from DB
            user_data = await self.external_api_service.get_user_data(user_id)

            if user_data:
                # Convert to dict if needed
                if hasattr(user_data, "model_dump"):
                    user_data_dict = user_data.model_dump()
                else:
                    user_data_dict = user_data

                # Add timestamp
                user_data_dict["last_updated"] = datetime.now().isoformat()

                # Save to general cache
                await self.redis_client.set_general_user_data(
                    user_id, user_data_dict, expire=self._cache_ttl
                )
                logger.info(f"✅ Created general cache from DB for user {user_id}")
                return True
            else:
                logger.info(f"ℹ️ No user data found in DB for user {user_id}")
                return False

        except Exception as e:
            logger.error(f"❌ Error ensuring general cache: {str(e)}")
            # Don't raise exception to avoid breaking the main flow, just report failure
            return False

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import redis
from config import settings

# Configure logging
logger = logging.getLogger(__name__)


class RedisClient:
    """Redis client for storing chat sessions and temporary data"""

    def __init__(self):
        self.redis = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password,
            # No username/password - Redis configured without authentication
            decode_responses=True,
            ssl_cert_reqs=None,  # Don't verify certificate (ElastiCache uses self-signed certs)
            socket_timeout=10.0,  # 4 second timeout for socket operations
            socket_connect_timeout=10.0,  # 4 second timeout for connection
            health_check_interval=60,  # Check connection health every 30 seconds
        )

    async def ping(self) -> bool:
        """Test Redis connection - non-blocking using executor"""
        try:
            # Run synchronous ping in executor so it doesn't block the event loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.redis.ping)
            logger.info(f"✅ Redis ping successful: {result}")
            return True
        except redis.ConnectionError as e:
            logger.error(f"❌ Redis connection error: {str(e)}")
            return False
        except redis.AuthenticationError as e:
            logger.error(f"❌ Redis authentication error: {str(e)}")
            return False
        except redis.TimeoutError as e:
            logger.error(f"❌ Redis timeout error: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"❌ Redis ping failed: {type(e).__name__}: {str(e)}")
            return False

    def _get_session_key(self, session_id: str) -> str:
        """Get Redis key for session (backend)"""
        return f"backend:session:{session_id}"

    def _get_pending_key(self, message_id: str) -> str:
        """Get Redis key for pending message"""
        return f"backend:pending:{message_id}"

    def _get_short_term_key(self, session_id: str) -> str:
        """Get Redis key for short-term memory (session-based)"""
        return f"chat:short_term:{session_id}"

    def _get_long_term_key(self, user_id: str) -> str:
        """Get Redis key for long-term memory (user-based)"""
        return f"chat:long_term:{user_id}"

    def _get_chat_session_key(self, session_id: str) -> str:
        """Get Redis key for chat session info"""
        return f"chat:session:{session_id}"

    def general_user_data_redis_key(self, user_id: str) -> str:
        """Redis key for cached full user / onboarding payload."""
        return f"general:{user_id}"

    def _get_general_key(self, user_id: str) -> str:
        return self.general_user_data_redis_key(user_id)

    async def store_pending_message(
        self, message_id: str, message_data: Dict[str, Any], ttl: int = 300  # 5 minutes
    ) -> None:
        """Store pending message waiting for response"""
        try:
            key = self._get_pending_key(message_id)
            self.redis.setex(key, ttl, json.dumps(message_data, default=str))
            logger.info(f"💾 Stored pending message {message_id}")
        except Exception as e:
            logger.error(f"❌ Error storing pending message: {str(e)}")

    async def get_pending_message(self, message_id: str) -> Optional[Dict[str, Any]]:
        """Get pending message data"""
        try:
            key = self._get_pending_key(message_id)
            data = self.redis.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"❌ Error getting pending message: {str(e)}")
            return None

    async def remove_pending_message(self, message_id: str) -> None:
        """Remove pending message after response received"""
        try:
            key = self._get_pending_key(message_id)
            self.redis.delete(key)
            logger.info(f"🗑️ Removed pending message {message_id}")
        except Exception as e:
            logger.error(f"❌ Error removing pending message: {str(e)}")

    async def save_chat_message(
        self,
        session_id: str,
        user_id: str,
        message: str,
        response: str,
        language: str,
        quick_reply: Optional[Dict[str, str]] = None,
        flow: str = "general",
    ) -> None:
        """Save a chat message and response to memory (short-term and long-term)

        Args:
            session_id: Session ID
            user_id: User ID
            message: User's message
            response: Bot's response
            quick_reply: Optional quick reply info
            flow: Flow type (general, onboarding, etc.)
        """
        try:
            timestamp = datetime.now().isoformat()
            if not message and quick_reply:
                message = quick_reply.get("label", "")

            chat_entry = {
                "timestamp": timestamp,
                "user_message": message,
                "bot_response": response,
                "flow": flow,
                "language": language,
            }

            # Use pipeline to batch Redis operations for better performance
            pipe = self.redis.pipeline()

            # Save to short-term memory (session-based)
            short_term_key = self._get_short_term_key(session_id)
            pipe.lpush(short_term_key, json.dumps(chat_entry))
            pipe.expire(short_term_key, 3600)  # 1 hour TTL
            logger.info(f"📤 Pushing to short-term: {short_term_key}")

            # Save to long-term memory (user-based) - keep only last 50 messages
            long_term_key = self._get_long_term_key(user_id)
            pipe.lpush(long_term_key, json.dumps(chat_entry))
            pipe.ltrim(long_term_key, 0, 49)  # Keep only last 50 messages
            pipe.expire(long_term_key, 86400 * 30)  # 30 days TTL
            logger.info(f"📤 Pushing to long-term: {long_term_key}")

            # Update session info
            session_key = self._get_chat_session_key(session_id)
            session_info = {
                "user_id": user_id,
                "flow": flow,
                "last_activity": timestamp,
                "message_count": 0,  # Will be updated after pipeline
            }
            pipe.hset(session_key, mapping=session_info)
            pipe.expire(session_key, 3600)  # 1 hour TTL

            # Execute all operations in one batch
            pipe.execute()

            logger.info(
                f"💾 Saved chat message for session {session_id}, user {user_id}"
            )
            logger.info(f"📝 User message: {message[:50]}...")
            logger.info(f"🤖 Bot response: {response[:50]}...")

        except Exception as e:
            logger.error(f"❌ Error saving chat message: {str(e)}")

    async def get_chat_history(
        self, session_id: str, user_id: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get chat history for session"""
        try:
            key = self._get_session_key(session_id)
            messages = self.redis.lrange(key, 0, limit - 1)

            chat_history = []
            for msg_json in messages:
                try:
                    msg_data = json.loads(msg_json)
                    chat_history.append(msg_data)
                except json.JSONDecodeError:
                    continue

            return chat_history
        except Exception as e:
            logger.error(f"❌ Error getting chat history: {str(e)}")
            return []

    async def get_chat_history_by_user_id(
        self,
        user_id: str,
        page: int = 1,
        limit: int = 20,
        sort_order: str = "desc",  # "asc" or "desc"
    ) -> Dict[str, Any]:
        """
        Get all chat history for a user_id from long-term memory
        with pagination and sorting by timestamp

        Args:
            user_id: The user ID to get history for
            page: Page number (1-indexed)
            limit: Number of messages per page
            sort_order: Sort order - "asc" (oldest first) or "desc" (newest first)

        Returns:
            Dict with messages, total, page, limit, total_pages
        """
        try:
            # Get long-term memory key for user
            long_term_key = self._get_long_term_key(user_id)

            # Get all messages from long-term memory
            messages_json = self.redis.lrange(long_term_key, 0, -1)  # Get all messages

            all_messages = []
            for msg_json in messages_json:
                try:
                    msg_data = json.loads(msg_json)
                    # Keep original format for UI display
                    # Format: {"timestamp": "...", "user_message": "...", "bot_response": "...", "flow": "..."}
                    all_messages.append(msg_data)
                except json.JSONDecodeError:
                    continue

            # Sort by timestamp
            reverse = sort_order.lower() == "desc"
            all_messages.sort(key=lambda x: x.get("timestamp", ""), reverse=reverse)

            # Pagination
            total = len(all_messages)
            total_pages = (
                (total + limit - 1) // limit if total > 0 else 0
            )  # Ceiling division
            start_idx = (page - 1) * limit
            end_idx = start_idx + limit
            paginated_messages = all_messages[start_idx:end_idx]

            logger.info(
                f"📚 Retrieved {len(paginated_messages)} messages for user {user_id} "
                f"(page {page}/{total_pages}, total: {total})"
            )

            return {
                "messages": paginated_messages,
                "total": total,
                "page": page,
                "limit": limit,
                "total_pages": total_pages,
            }

        except Exception as e:
            logger.error(f"❌ Error getting chat history by user_id: {str(e)}")
            return {
                "messages": [],
                "total": 0,
                "page": page,
                "limit": limit,
                "total_pages": 0,
            }

    async def clear_chat_session(self, session_id: str) -> None:
        """Clear chat session (short-term memory and session info)"""
        try:
            short_term_key = self._get_short_term_key(session_id)
            session_key = self._get_chat_session_key(session_id)
            self.redis.delete(short_term_key)
            self.redis.delete(session_key)
            logger.info(f"🗑️ Cleared chat session {session_id}")
        except Exception as e:
            logger.error(f"❌ Error clearing chat session: {str(e)}")

    async def set_data(self, key: str, data: Any, expire: Optional[int] = 86400) -> None:
        """Store arbitrary data in Redis. Pass expire=None to disable TTL."""
        try:
            if expire is None:
                self.redis.set(key, json.dumps(data, default=str))
            else:
                self.redis.setex(key, expire, json.dumps(data, default=str))
            logger.info(f"💾 Stored data with key {key}")
        except Exception as e:
            logger.error(f"❌ Error storing data: {str(e)}")

    async def get_data(self, key: str) -> Optional[Any]:
        """Get arbitrary data from Redis"""
        try:
            data = self.redis.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"❌ Error getting data: {str(e)}")
            return None

    async def delete(self, key: str) -> None:
        """Delete a key from Redis"""
        try:
            self.redis.delete(key)
        except Exception as e:
            logger.error(f"❌ Error deleting key {key}: {str(e)}")

    async def get_general_user_data(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get general user data from cache"""
        try:
            general_key = self._get_general_key(user_id)
            return await self.get_data(general_key)
        except Exception as e:
            logger.error(f"❌ Error getting general user data: {str(e)}")
            return None

    async def set_general_user_data(
        self, user_id: str, user_data: Dict[str, Any], expire: int = 86400 * 30
    ) -> None:
        """Set general user data in cache"""
        try:
            general_key = self._get_general_key(user_id)
            await self.set_data(general_key, user_data, expire=expire)
        except Exception as e:
            logger.error(f"❌ Error setting general user data: {str(e)}")

    async def get_profile_id_from_cache(self, user_id: str) -> Optional[str]:
        """Get profile_id from general user data cache"""
        try:
            general_data = await self.get_general_user_data(user_id)
            if general_data and isinstance(general_data, dict):
                return general_data.get("profile_id")
            return None
        except Exception as e:
            logger.error(f"❌ Error getting profile_id from cache: {str(e)}")
            return None

    def _get_insight_cache_key(
        self, user_id: str, insight_type: str, language: str = "en-US"
    ) -> str:
        """Get Redis key for cached insight

        Format matches productivity_insight structure: {insight_type}_insight:{user_id}:{language}
        Examples:
            - need_attention_insight:fa3ee83e-94cb-4199-986b-adb014c05e64:en-US
            - great_job_insight:fa3ee83e-94cb-4199-986b-adb014c05e64:en-US
            - opportunity_insight:fa3ee83e-94cb-4199-986b-adb014c05e64:vi-VN

        This creates a folder-like structure in Redis (using : as separator):
            need_attention_insight/
                └── fa3ee83e-94cb-4199-986b-adb014c05e64/
                    └── en-US (string data)

        Args:
            user_id: User ID
            insight_type: Type of insight (need-attention, great-job, opportunity)
            language: Language code (e.g., en-US, vi-VN)

        Returns:
            Redis key string
        """
        # Convert insight_type from kebab-case to snake_case for consistency
        # need-attention -> need_attention, great-job -> great_job, opportunity -> opportunity
        insight_type_snake = insight_type.replace("-", "_")
        return f"{insight_type_snake}_insight:{user_id}:{language}"

    async def get_chat_history_timestamp(self, user_id: str) -> Optional[str]:
        """Get the timestamp of the most recent chat message for a user

        Args:
            user_id: User ID

        Returns:
            ISO timestamp string of most recent message, or None if no messages
        """
        try:
            long_term_key = self._get_long_term_key(user_id)
            # Get the most recent message (index 0)
            messages_json = self.redis.lrange(long_term_key, 0, 0)

            if messages_json:
                msg_data = json.loads(messages_json[0])
                timestamp = msg_data.get("timestamp")
                logger.info(f"📅 Latest chat timestamp for user {user_id}: {timestamp}")
                return timestamp

            logger.info(f"📅 No chat history found for user {user_id}")
            return None

        except Exception as e:
            logger.error(f"❌ Error getting chat history timestamp: {str(e)}")
            return None

    async def cache_insight(
        self,
        user_id: str,
        insight_type: str,
        insight_data: Dict[str, Any],
        chat_history_timestamp: Optional[str],
        language: str = "en-US",
        ttl: int = 3600,  # 1 hour default
    ) -> None:
        """Cache insight with chat history timestamp for smart invalidation

        Args:
            user_id: User ID
            insight_type: Type of insight (need-attention, great-job, opportunity)
            insight_data: Insight data to cache
            chat_history_timestamp: Timestamp of chat history when insight was generated
            language: Language code (default: en-US)
            ttl: Time to live in seconds (default: 1 hour)
        """
        try:
            cache_key = self._get_insight_cache_key(user_id, insight_type, language)

            # Store insight with metadata
            cache_payload = {
                "insight_data": insight_data,
                "chat_history_timestamp": chat_history_timestamp,
                "cached_at": json.dumps(datetime.now().isoformat(), default=str),
            }

            self.redis.setex(cache_key, ttl, json.dumps(cache_payload, default=str))
            logger.info(
                f"💾 Cached {insight_type} insight for user {user_id} (lang: {language}) "
                f"(chat_ts: {chat_history_timestamp})"
            )

        except Exception as e:
            logger.error(f"❌ Error caching insight: {str(e)}")

    async def get_cached_insight(
        self, user_id: str, insight_type: str, language: str = "en-US"
    ) -> Optional[Dict[str, Any]]:
        """Get cached insight if valid (chat history hasn't changed)

        Args:
            user_id: User ID
            insight_type: Type of insight (need-attention, great-job, opportunity)
            language: Language code (default: en-US)

        Returns:
            Cached insight data if valid, None if cache miss or invalidated
        """
        try:
            cache_key = self._get_insight_cache_key(user_id, insight_type, language)
            cached_data = self.redis.get(cache_key)

            if not cached_data:
                logger.info(f"🔍 No cached {insight_type} insight for user {user_id}")
                return None

            cache_payload = json.loads(cached_data)
            cached_chat_timestamp = cache_payload.get("chat_history_timestamp")
            insight_data = cache_payload.get("insight_data")

            # Get current chat history timestamp
            current_chat_timestamp = await self.get_chat_history_timestamp(user_id)

            # Validate cache: if chat history changed, invalidate
            if cached_chat_timestamp != current_chat_timestamp:
                logger.info(
                    f"🔄 Cache invalidated for {insight_type} (user {user_id}): "
                    f"Chat history changed from {cached_chat_timestamp} to {current_chat_timestamp}"
                )
                # Delete stale cache
                self.redis.delete(cache_key)
                return None

            logger.info(
                f"✅ Valid cached {insight_type} insight found for user {user_id} "
                f"(chat_ts matches: {current_chat_timestamp})"
            )
            return insight_data

        except Exception as e:
            logger.error(f"❌ Error getting cached insight: {str(e)}")
            return None

    async def invalidate_insight_cache(self, user_id: str) -> None:
        """Invalidate all cached insights for a user (called when chat history changes)

        Uses Redis pattern matching to delete ALL language variants automatically.

        Args:
            user_id: User ID
        """
        try:
            insight_types = ["need_attention", "great_job", "opportunity"]
            deleted_count = 0

            for insight_type in insight_types:
                # Pattern matches ANY language: need_attention_insight:{user_id}:*
                pattern = f"{insight_type}_insight:{user_id}:*"

                # Find all keys matching pattern (works for any language: en-US, vi-VN, ja-JP, etc.)
                matching_keys = self.redis.keys(pattern)

                # Delete all found keys
                if matching_keys:
                    deleted = self.redis.delete(*matching_keys)
                    deleted_count += deleted
                    logger.info(
                        f"🗑️ Deleted {deleted} cache(s) for {insight_type} (pattern: {pattern})"
                    )

            logger.info(
                f"🗑️ Invalidated {deleted_count} insight cache(s) for user {user_id} (all languages)"
            )

        except Exception as e:
            logger.error(f"❌ Error invalidating insight cache: {str(e)}")

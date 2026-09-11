"""Manager for pending action cache"""

import json
from datetime import datetime
from typing import Any, Dict, Optional

from app.utils.logger import logger
from app.utils.redis_client import get_redis_client

# Redis client
redis_client = get_redis_client()


class PendingActionManager:
    """Manager for pending action cache operations"""

    CACHE_TTL = 1800  # 30 minutes

    @staticmethod
    def _get_cache_key(user_id: str, session_id: str, tool_name: str) -> str:
        """Generate cache key for pending action"""
        return f"pending_action:{user_id}:{session_id}:{tool_name}"

    def save(
        self, user_id: str, session_id: str, tool_name: str, data: Dict[str, Any]
    ) -> bool:
        """Save pending action data to cache

        Args:
            user_id: User ID
            session_id: Session ID for session-specific cache
            tool_name: Tool name (e.g., "update_health_conditions")
            data: Data to save (should include params, body, status, etc.)

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            key = self._get_cache_key(user_id, session_id, tool_name)

            # Add metadata
            cache_data = {
                **data,
                "user_id": user_id,
                "tool_name": tool_name,
                "timestamp": datetime.now().isoformat(),
            }

            redis_client.set(
                key,
                json.dumps(cache_data, separators=(",", ":"), ensure_ascii=False),
                ex=self.CACHE_TTL,
            )
            logger.info(
                f"✅ Saved pending action to cache: {key} (status: {data.get('status', 'unknown')})"
            )
            return True
        except Exception as e:
            logger.error(f"❌ Error saving pending action to cache: {str(e)}")
            return False

    def get(
        self, user_id: str, session_id: str, tool_name: str
    ) -> Optional[Dict[str, Any]]:
        """Get pending action data from cache

        Args:
            user_id: User ID
            session_id: Session ID for session-specific cache
            tool_name: Tool name

        Returns:
            Cached data dict or None if not found
        """
        try:
            key = self._get_cache_key(user_id, session_id, tool_name)
            data = redis_client.get(key)

            if data:
                result = json.loads(data) if isinstance(data, str) else data
                logger.info(f"✅ Retrieved pending action from cache: {key}")
                return result

            logger.debug(f"ℹ️ No pending action found in cache: {key}")
            return None
        except Exception as e:
            logger.error(f"❌ Error getting pending action from cache: {str(e)}")
            return None

    def delete(self, user_id: str, session_id: str, tool_name: str) -> bool:
        """Delete pending action from cache

        Args:
            user_id: User ID
            session_id: Session ID for session-specific cache
            tool_name: Tool name

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            key = self._get_cache_key(user_id, session_id, tool_name)
            redis_client.delete(key)
            logger.info(f"✅ Deleted pending action from cache: {key}")
            return True
        except Exception as e:
            logger.error(f"❌ Error deleting pending action from cache: {str(e)}")
            return False

    def extend_ttl(self, user_id: str, session_id: str, tool_name: str) -> bool:
        """Extend TTL for existing pending action

        Args:
            user_id: User ID
            session_id: Session ID for session-specific cache
            tool_name: Tool name

        Returns:
            True if extended successfully, False otherwise
        """
        try:
            key = self._get_cache_key(user_id, session_id, tool_name)
            redis_client.expire(key, self.CACHE_TTL)
            logger.info(f"✅ Extended TTL for pending action: {key}")
            return True
        except Exception as e:
            logger.error(f"❌ Error extending TTL for pending action: {str(e)}")
            return False

    # ============================================================
    # Sync Cache Methods - For syncing events to Google Calendar
    # ============================================================

    @staticmethod
    def _get_sync_cache_key(user_id: str, session_id: str) -> str:
        """Generate cache key for sync pending"""
        return f"sync_pending:{user_id}:{session_id}"

    def save_sync_pending(
        self,
        user_id: str,
        session_id: str,
        event_ids: list,
        token: str = None,
        event_payload: Optional[Dict[str, Any]] = None,
        provider_name: Optional[str] = None,
        is_single_event: bool = False,
        is_online_meeting: bool = False,
    ) -> bool:
        """Save event IDs pending for sync to Google

        Args:
            user_id: User ID
            session_id: Session ID
            event_ids: List of event IDs to sync
            token: JWT token for API calls
            event_payload: Full event payload used for post-sync operations (single event only)
            provider_name: Event provider name for update API (default: insidesync)
            is_single_event: Whether this sync payload comes from single-event create flow

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            key = self._get_sync_cache_key(user_id, session_id)

            cache_data = {
                "event_ids": event_ids,
                "token": token,
                "event_payload": event_payload,
                "provider_name": provider_name,
                "is_single_event": is_single_event,
                "is_online_meeting": is_online_meeting,
                "user_id": user_id,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
            }

            redis_client.set(
                key,
                json.dumps(
                    cache_data, separators=(",", ":"), ensure_ascii=False, default=str
                ),
                ex=self.CACHE_TTL,
            )
            logger.info(
                f"✅ Saved sync pending to cache: {key} ({len(event_ids)} events)"
            )
            return True
        except Exception as e:
            logger.error(f"❌ Error saving sync pending to cache: {str(e)}")
            return False

    def get_sync_pending(
        self, user_id: str, session_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get sync pending data from cache

        Args:
            user_id: User ID
            session_id: Session ID

        Returns:
            Cached data dict or None if not found
        """
        try:
            key = self._get_sync_cache_key(user_id, session_id)
            data = redis_client.get(key)

            if data:
                result = json.loads(data) if isinstance(data, str) else data
                logger.info(f"✅ Retrieved sync pending from cache: {key}")
                return result

            logger.debug(f"ℹ️ No sync pending found in cache: {key}")
            return None
        except Exception as e:
            logger.error(f"❌ Error getting sync pending from cache: {str(e)}")
            return None

    def delete_sync_pending(self, user_id: str, session_id: str) -> bool:
        """Delete sync pending from cache

        Args:
            user_id: User ID
            session_id: Session ID

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            key = self._get_sync_cache_key(user_id, session_id)
            redis_client.delete(key)
            logger.info(f"✅ Deleted sync pending from cache: {key}")
            return True
        except Exception as e:
            logger.error(f"❌ Error deleting sync pending from cache: {str(e)}")
            return False

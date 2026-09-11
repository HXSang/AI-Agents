"""Redis store for multi-action queues"""

import asyncio
from typing import Any, Dict, List, Optional

from app.services.memory_service import IMemoryService
from app.utils.logger import logger

_ACTION_QUEUE_TTL = 1800


def _get_draft_key(user_id: str, session_id: str) -> str:
    return f"user:{user_id}:session:{session_id}:multi_action:draft"


async def save_draft_queue(
    user_id: str,
    session_id: str,
    action_queue: List[Dict[str, Any]],
    memory_service: IMemoryService,
    original_message: str = "",
) -> None:
    """Saves a draft while awaiting user confirmation (not yet added to pending)."""
    key = _get_draft_key(user_id, session_id)
    payload = {
        "action_queue": action_queue,
        "original_message": original_message,
    }
    await memory_service.set_json(key, payload, ttl=_ACTION_QUEUE_TTL)
    logger.info(f"📝 Saved DRAFT to Redis -> {len(action_queue)} actions")


async def get_and_clear_draft_queue(
    user_id: str,
    session_id: str,
    memory_service: IMemoryService,
) -> Optional[Dict[str, Any]]:
    """Reads the draft and auto-deletes it for cleanup."""
    key = _get_draft_key(user_id, session_id)
    data = await memory_service.get_json(key)
    if data:
        await memory_service.delete_key(key)
        logger.info("📥 Loaded and cleared DRAFT from Redis")
        return data
    return None


async def get_draft_queue(
    user_id: str,
    session_id: str,
    memory_service: IMemoryService,
) -> Optional[Dict[str, Any]]:
    """Peek the draft WITHOUT deleting it.

    Used by the free-text edit flow to inspect a pending plan before deciding
    whether the user's message edits it. The draft stays in Redis so a later
    Continue/Cancel still works.
    """
    key = _get_draft_key(user_id, session_id)
    return await memory_service.get_json(key)


def _get_pending_key(user_id: str, session_id: str) -> str:
    return f"user:{user_id}:session:{session_id}:multi_action:pending"


def _get_completed_key(user_id: str, session_id: str) -> str:
    return f"user:{user_id}:session:{session_id}:multi_action:completed"


async def save_action_queues(
    user_id: str,
    session_id: str,
    action_queue: List[Dict[str, Any]],
    complete_action_queue: List[Dict[str, Any]],
    memory_service: IMemoryService,
    original_message: str = "",
) -> None:
    pending_key = _get_pending_key(user_id, session_id)
    completed_key = _get_completed_key(user_id, session_id)

    pending_payload = {
        "action_queue": action_queue,
        "original_message": original_message,
    }

    completed_payload = {
        "complete_action_queue": complete_action_queue,
    }

    await asyncio.gather(
        memory_service.set_json(pending_key, pending_payload, ttl=_ACTION_QUEUE_TTL),
        memory_service.set_json(
            completed_key, completed_payload, ttl=_ACTION_QUEUE_TTL
        ),
    )

    logger.info(
        f"💾 Saved cleanly to Redis -> Pending: {len(action_queue)} | Completed: {len(complete_action_queue)}"
    )


async def get_action_queues(
    user_id: str,
    session_id: str,
    memory_service: IMemoryService,
) -> Optional[Dict[str, Any]]:
    pending_key = _get_pending_key(user_id, session_id)
    completed_key = _get_completed_key(user_id, session_id)

    pending_data, completed_data = await asyncio.gather(
        memory_service.get_json(pending_key), memory_service.get_json(completed_key)
    )

    if pending_data or completed_data:
        combined_data = {
            "action_queue": (pending_data or {}).get("action_queue", []),
            "original_message": (pending_data or {}).get("original_message", ""),
            "complete_action_queue": (completed_data or {}).get(
                "complete_action_queue", []
            ),
        }
        logger.info("📥 Loaded cleanly from Redis")
        return combined_data

    return None


async def delete_action_queues(
    user_id: str,
    session_id: str,
    memory_service: IMemoryService,
) -> None:
    pending_key = _get_pending_key(user_id, session_id)
    completed_key = _get_completed_key(user_id, session_id)

    await asyncio.gather(
        memory_service.delete_key(pending_key), memory_service.delete_key(completed_key)
    )
    logger.info(f"🗑️ Cleaned up multi-action keys")

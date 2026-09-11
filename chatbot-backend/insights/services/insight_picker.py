from __future__ import annotations

import random
from typing import Optional

from clients.redis_client import RedisClient
from insights.schemas.insight_item import InsightItem
from insights.schemas.insight_pool import InsightPool
from utils.logger import logger

POOL_KEY = "insight_pool:v2:{user_id}"
SEEN_KEY = "insight_seen:v2:{user_id}"


async def _load_pool(redis: RedisClient, user_id: str) -> Optional[InsightPool]:
    """Load InsightPool from Redis. Returns None if not found or expired."""
    key = POOL_KEY.format(user_id=user_id)
    cached = await redis.get_data(key)
    if not cached or "pool" not in cached:
        return None
    try:
        return InsightPool.model_validate_json(cached["pool"])
    except Exception as e:
        logger.warning(f"Failed to parse InsightPool for {user_id}: {e}")
        return None


async def _save_pool(redis: RedisClient, user_id: str, pool: InsightPool) -> None:
    """Save InsightPool to Redis with TTL."""
    key = POOL_KEY.format(user_id=user_id)
    await redis.set_data(key, {"pool": pool.model_dump_json()}, expire=None)


async def _get_seen_ids(redis: RedisClient, user_id: str) -> set[str]:
    """Load the set of insight_ids already shown to the user."""
    key = SEEN_KEY.format(user_id=user_id)
    cached = await redis.get_data(key)
    if cached and "seen_ids" in cached:
        return set[str](cached["seen_ids"])
    return set[str]()


async def _mark_seen(redis: RedisClient, user_id: str, insight_id: str) -> None:
    """Mark an insight_id as seen, with TTL matching the pool."""
    key = SEEN_KEY.format(user_id=user_id)
    seen_ids = await _get_seen_ids(redis, user_id)
    seen_ids.add(insight_id)
    await redis.set_data(key, {"seen_ids": list(seen_ids)}, expire=None)


async def _reset_seen(redis: RedisClient, user_id: str) -> None:
    """Reset seen history — used when pool is exhausted or data changes."""
    key = SEEN_KEY.format(user_id=user_id)
    await redis.delete(key)


async def pick_random_insight(
    user_id: str,
    domain: str,
    redis: RedisClient,
) -> Optional[InsightItem]:
    pool = await _load_pool(redis, user_id)
    if not pool:
        logger.info(f"No InsightPool found for {user_id} — pool not yet generated")
        return None

    candidates = pool.by_domain(domain)
    if not candidates:
        logger.info(f"No {domain} insights in pool for {user_id}")
        return None

    seen_ids = await _get_seen_ids(redis, user_id)
    unseen = [c for c in candidates if c.insight_id not in seen_ids]

    if unseen:
        # Normal case: pick from unseen
        chosen = random.choice(unseen)
    else:
        # Pool exhausted for this domain — reset and pick fresh
        logger.info(f"{domain} pool exhausted for {user_id} — resetting seen history")
        await _reset_seen(redis, user_id)
        chosen = random.choice(candidates)

    await _mark_seen(redis, user_id, chosen.insight_id)
    logger.info(
        f"Picked insight {chosen.insight_id[:8]} ({domain}) for {user_id}"
    )
    return chosen


async def clear_insight_history(user_id: str, redis: RedisClient) -> None:
    pool_key = POOL_KEY.format(user_id=user_id)
    seen_key = SEEN_KEY.format(user_id=user_id)
    await redis.delete(pool_key)
    await redis.delete(seen_key)
    logger.info(f"Cleared insight pool and history for {user_id}")


async def pool_exists(user_id: str, redis: RedisClient) -> bool:
    """Check if an InsightPool exists for this user."""
    key = POOL_KEY.format(user_id=user_id)
    cached = await redis.get_data(key)
    return cached is not None
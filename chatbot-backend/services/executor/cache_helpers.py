"""Shared cache helpers for executor services (Redis + external API patterns)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from services.external_api_service import IExternalAPIService
from utils.logger import logger


class CacheHelpers:
    """Stateless cache helpers shared across executor services.

    Grouped as ``@staticmethod`` to follow the ``services/executor/`` folder
    convention (sibling files ``constant.py``, ``prepare_*.py`` are also
    class-wrapped). Has no per-instance state — call directly on the class:
    ``CacheHelpers.financial_insight_redis_key(...)``.
    """

    @staticmethod
    def financial_insight_redis_key(user_id: str, language: str) -> str:
        """Redis key for cached financial insight payload (per user, YYYY-MM month, locale)."""
        return f"financial_insight:{user_id}:{language}"

    @staticmethod
    def monthly_insight_redis_key(profile_id: str, year: int, month: int) -> str:
        """Redis key for monthly insight payload (per profile + period).

        Holds the rendered top-N insights ({year, month, insights}) as JSON.
        Used for anti-repetition novelty lookup across the 3-month window.
        """
        return f"monthly_insight:{profile_id}:{year:04d}{month:02d}"

    @staticmethod
    def monthly_snapshot_redis_key(profile_id: str, year: int, month: int) -> str:
        """Redis key for monthly snapshot (raw `MonthlySnapshot` JSON).

        Persists the computed monthly rollup so subsequent months can perform
        cross-month comparisons per Rules §1 Phase 1 rows 1-3 / 9-11,
        Phase 2 rows 3 / 12-13, Phase 3 rows 10-13. Lookback window matches
        the anti-rep window so we can cover 90d / 180d baselines.
        """
        return f"monthly_snapshot:{profile_id}:{year:04d}{month:02d}"

    @staticmethod
    def monthly_insight_view_redis_key(
        profile_id: str, year: int, month: int, language: str
    ) -> str:
        """Redis key for the rendered+translated monthly insight VIEW of a
        COMPLETED past month.

        A finished month's daily data and baseline are immutable, so its final
        response is deterministic and can be served verbatim on repeat calls
        without re-running detection / LLM render / translation. Keyed by
        ``language`` since the payload is already translated.
        """
        return f"monthly_insight_view:{profile_id}:{year:04d}{month:02d}:{language}"

    @staticmethod
    async def fetch_general_user_data_cached(
        redis_client: Any,
        external_api_service: Optional[IExternalAPIService],
        user_id: str,
        *,
        log_prefix: str = "User profile",
    ) -> Optional[Dict[str, Any]]:
        try:
            if not external_api_service:
                return None
            user_data = await external_api_service.get_user_data(user_id)
            if not user_data:
                return None
            if hasattr(user_data, "model_dump"):
                user_data_dict = user_data.model_dump()
            else:
                user_data_dict = user_data
            if not isinstance(user_data_dict, dict):
                return None
            return dict(user_data_dict)
        except Exception as e:
            logger.warning(
                f"⚠️ {log_prefix}: Failed to load general user data for user {user_id}: {e}"
            )
            return None

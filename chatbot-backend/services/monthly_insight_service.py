"""Monthly Insight Service Interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from models.models import MonthlyInsightResponse


class MonthlyInsightService(ABC):
    """Abstract interface for the monthly insight pipeline."""

    @abstractmethod
    async def generate_monthly(
        self,
        profile_id: str,
        year: Optional[int] = None,
        month: Optional[int] = None,
        force_update: bool = False,
        language: Optional[str] = "en-US",
        timezone: Optional[str] = None,
    ) -> MonthlyInsightResponse:
        """Run the full monthly insight pipeline.

        Resolves the target month internally: an explicit ``year``+``month`` is
        honoured, otherwise the current month in ``timezone`` is used. On days
        1-2 a current-month target is redirected to the previous month so the
        month-end synthesis has a complete month of data.

        Args:
            profile_id: Profile UUID
            year: Analysis year (optional — defaults to current year in tz)
            month: Analysis month 1..12 (optional — defaults to current month)
            force_update: Reserved for future cache layer; currently no-op
            language: IETF language tag (e.g. "en-US", "vi-VN") controlling LLM
                output language. Mapping in ``agents.prompt.get_language_name``.
                Falls back to English templates if LLM call fails.
            timezone: Optional IANA TZ name (e.g. "Asia/Ho_Chi_Minh") forwarded
                to the snapshot API for day-bucketing.

        Returns:
            MonthlyInsightResponse with status='success' on happy path, or
            status='error' when the upstream snapshot API is unreachable.
        """
        pass

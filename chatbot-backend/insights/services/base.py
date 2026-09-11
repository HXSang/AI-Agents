"""Insight Service Interface"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class InsightService(ABC):
    """Abstract interface for insight services"""

    @abstractmethod
    async def analyze_productivity(
        self,
        user_id: str,
        timezone: Optional[str] = None,
        provider_name: Optional[str] = "",
        language: Optional[str] = "en-US",
        force_update: Optional[bool] = False,
    ) -> Dict[str, Any]:
        """
        Analyze user productivity based on today's calendar data.

        Args:
            user_id: User ID to analyze
            timezone: Optional IANA timezone
            provider_name: Optional calendar provider name (default: empty string)
            language: Optional language code (default: "en")
            force_update: Optional boolean (default: False). If True, bypass cache and recompute insight

        Returns:
            Dict with insight analysis result containing:
            - status: "success" or "error"
            - user_id: User ID
            - insight: AI-generated productivity analysis (if success)
            - error: Error message (if error)
        """
        pass

    @abstractmethod
    async def analyze_overall_insight(
        self,
        user_id: str,
        timezone: Optional[str] = None,
        provider_name: Optional[str] = "",
        language: Optional[str] = "en-US",
        force_update: Optional[bool] = False,
    ) -> Dict[str, Any]:
        """
        Analyze overall insights combining great_job, need_attention, and opportunity.

        Args:
            user_id: User ID to analyze
            timezone: Optional IANA timezone
            provider_name: Optional calendar provider name (default: empty string)
            language: Optional language code (default: "en-US")
            force_update: Optional boolean (default: False). If True, bypass cache and recompute insight

        Returns:
            Dict with overall insight analysis result containing:
            - status: "success" or "error"
            - user_id: User ID
            - insight: Dict with great_job, need_attention, opportunity (if success)
            - error: Error message (if error)
        """
        pass

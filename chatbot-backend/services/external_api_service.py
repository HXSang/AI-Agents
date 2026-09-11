"""External API Service for integrating with backend application"""

from abc import ABC, abstractmethod
from datetime import date
from typing import Any, Dict, List, Optional

from models.models import BalanceScoreDTO, DailySnapshot, HealthSummary


class IExternalAPIService(ABC):
    """Interface for External API Service"""

    @abstractmethod
    async def login_and_get_token(self) -> Optional[str]:
        """Login to external API and get bearer token"""
        pass

    @abstractmethod
    async def save_onboarding_data(self, user_data: Dict[str, Any]) -> bool:
        """Save onboarding data to external API"""
        pass

    @abstractmethod
    async def get_user_data(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user data from database via external API"""
        pass

    @abstractmethod
    async def get_calendar_events(
        self,
        user_id: str,
        provider_name: Optional[str] = "",
        timezone: Optional[str] = None,
        start_date_time: Optional[str] = None,  # Format: "YYYY-MM-DDTHH:mm"
        end_date_time: Optional[str] = None,  # Format: "YYYY-MM-DDTHH:mm"
        include_cancelled: Optional[bool] = False,
        include_declined: Optional[bool] = False,
    ) -> Optional[List[Dict[str, Any]]]:
        """Get calendar events from external API within date range.

        Args:
            user_id: User ID
            provider_name: Optional provider name (default: empty string)
            timezone: Optional IANA timezone
            start_date_time: Optional start datetime (format: "YYYY-MM-DDTHH:mm")
            end_date_time: Optional end datetime (format: "YYYY-MM-DDTHH:mm")
            include_cancelled: Include cancelled events (default: False)
            include_declined: Include declined events (default: False)

        Returns:
            List of calendar events or None if error
        """
        pass

    @abstractmethod
    async def get_reminders(
        self,
        user_id: str,
        start_date: str,
        end_date: str,
        timezone: Optional[str] = None,
        provider_name: Optional[str] = None,
        incomplete_only: bool = False,
        page: int = 1,
        size: int = 50,
    ) -> Optional[List[Dict[str, Any]]]:
        pass

    @abstractmethod
    async def get_work_hours(
        self,
        user_id: str,
        date: str,
        timezone: Optional[str] = None,
        provider_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    async def get_reminder_lists(
        self, user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def get_reminder_by_id(
        self,
        reminder_id: str,
        user_id: Optional[str] = None,
        original_due_date: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    async def get_calendar_event_by_id(
        self,
        event_id: str,
        user_id: Optional[str] = None,
        original_start_time: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    async def get_calendar_events_today(
        self,
        user_id: Optional[str] = None,
        profile_id: Optional[str] = None,
        provider_name: Optional[str] = None,
        timezone: Optional[str] = None,
        include_cancelled: Optional[bool] = False,
        include_declined: Optional[bool] = False,
    ) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def get_calendar_sync_status(
        self,
        user_id: Optional[str] = None,
        profile_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def get_profile_id(self, user_id: str) -> Optional[str]:
        """Get profile_id for a user

        Args:
            user_id: User ID

        Returns:
            profile_id if found, None otherwise
        """
        pass

    @abstractmethod
    async def save_conversation(self, user_id: str, payload: Dict[str, Any]) -> bool:
        """Save conversation to database via external API

        Args:
            user_id: User ID
            payload: Conversation payload with profileId, timestamp, userMessage, botResponse, flow

        Returns:
            True if saved successfully, False otherwise
        """

    async def get_user_profiles(self, user_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get user profiles from external API (returns list)"""
        pass

    @abstractmethod
    async def get_productivity_summary(
        self,
        user_id: str,
        date: Optional[str] = None,
        timezone: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    async def get_productivity_summaries_by_range(
        self,
        user_id: str,
        start_date: str,
        end_date: str,
        timezone: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get daily productivity summaries by date range from /api/productivity/summaries."""
        pass

    @abstractmethod
    async def sync_emails(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Trigger email sync for user"""
        pass

    @abstractmethod
    async def check_email_sync_status(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Check email sync status for user"""
        pass

    @abstractmethod
    async def get_latest_mood(
        self,
        user_id: str,
        target_date: Optional[str] = None,
        timezone: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    async def get_moods(
        self,
        user_id: Optional[str] = None,
        page: int = 1,
        size: int = 20,
        start_datetime: Optional[str] = None,
        end_datetime: Optional[str] = None,
        timezone: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    async def get_health_sample_types(
        self, user_id: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Get available health sample types for a user from external API"""
        pass

    @abstractmethod
    async def get_all_health_stats_dynamic(
        self, user_id: str, timezone: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get all health stats dynamically for a user using two-step API pattern"""
        pass

    @abstractmethod
    async def get_health_summaries(
        self,
        user_id: str,
        type_code: Optional[str] = None,
        timezone: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get health summaries from /api/health/summaries endpoint.

        Args:
            user_id: User ID
            type_code: Optional type code (HR, SLEEP, STEPS, ENERGY). If None, returns all 4 types
            timezone: Optional timezone for date-range calculation.
                Fetches from (today - 7 days) to today (inclusive) in the given timezone.

        Returns:
            - If type_code provided: Single summary dict
            - If type_code is None: Dict with all 4 summaries keyed by type
        """
        pass

    @abstractmethod
    async def get_health_summaries_by_range(
        self,
        user_id: str,
        start_date: str,
        end_date: str,
        timezone: Optional[str] = None,
    ) -> Optional[List[HealthSummary]]:
        """Get health summaries theo date range từ /api/health/summaries (tất cả types)

        Args:
            user_id: User ID
            start_date: Start date in ISO format (e.g., "2026-01-25T17:00:00Z")
            end_date: End date in ISO format (e.g., "2026-02-01T16:59:59.999Z")
            timezone: Optional timezone

        Returns:
            List of HealthSummary models (EnergyHealthSummary, HRHealthSummary, SleepHealthSummary, StepsHealthSummary)
            Mỗi model có field "type" để identify type_code
        """
        pass

    @abstractmethod
    async def get_health_summary_latest_by_type(
        self,
        summary_type: str,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    async def get_health_stats_daily(
        self,
        sample_type: str,
        start_date: str,
        end_date: str,
        user_id: Optional[str] = None,
        timezone: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def get_sleep_stages_history(
        self,
        user_id: str,
        start_date: str,
        end_date: str,
        timezone: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch multi-day sleep stage breakdown (deep / rem / core) for a date range.

        Reuses the existing ``/api/health/summaries`` endpoint: calls
        ``get_health_summaries_by_range`` and extracts ``deep``, ``rem``, and
        ``core`` (hours) from each ``SleepDataEntry`` in the SLEEP summary's
        ``data`` list.

        Args:
            user_id: User ID.
            start_date: Inclusive start as ``YYYY-MM-DD``.
            end_date: Inclusive end as ``YYYY-MM-DD``.
            timezone: Optional IANA timezone.

        Returns:
            List of dicts, each containing at minimum:
            ``{"date": str, "deep_min": int, "rem_min": int, "light_min": int}``.
            Returns an empty list if the SLEEP summary is unavailable.
        """
        pass

    @abstractmethod
    async def get_finance_summary(
        self, user_id: str, month: str
    ) -> Optional[Dict[str, Any]]:
        """Get finance summary for a user/month from external API."""
        pass

    @abstractmethod
    async def get_balance_score(
        self, user_id: str, date: str, timezone: Optional[str] = None
    ) -> Optional[BalanceScoreDTO]:
        """Get balance score (productivity/finance/balance) by date."""
        pass

    @abstractmethod
    async def get_balance_scores_by_range(
        self,
        user_id: str,
        start_date: str,
        end_date: str,
        timezone: Optional[str] = None,
    ) -> List[BalanceScoreDTO]:
        """Get daily balance scores by date range from /api/balance."""
        pass

    @abstractmethod
    async def get_balance_streak(
        self,
        user_id: Optional[str] = None,
        timezone: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    async def get_finance_goals(self, user_id: str) -> List[Dict[str, Any]]:
        """Get finance goals for a user from external API."""
        pass

    @abstractmethod
    async def get_finance_bills(self, user_id: str) -> List[Dict[str, Any]]:
        """Get finance bills for a user from external API."""
        pass

    @abstractmethod
    async def get_finance_logs(
        self,
        user_id: str,
        month: Optional[str] = None,
        log_type: Optional[str] = None,
        categories: Optional[List[str]] = None,
        page: int = 1,
        size: int = 100,
    ) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def get_finance_budgets(
        self,
        user_id: str,
        budget_type: Optional[str] = None,
        categories: Optional[List[str]] = None,
        active: bool = True,
    ) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def get_daily_snapshots(
        self,
        profile_id: str,
        start: date,
        end: date,
        timezone: Optional[str] = None,
        language: Optional[str] = None,
    ) -> List[DailySnapshot]:
        """Fetch daily snapshot rows from BE pipeline for ``[start, end]`` inclusive.

        Hits the snapshot pipeline service (separate from main app API),
        configured via ``BE_SNAPSHOT_API_*`` env vars. Used by the Monthly
        Insight pipeline.

        Endpoint::

            GET {BE_SNAPSHOT_API_BASE_URL}/api/pipeline/daily-user-snapshots
                ?userId=<id>&startDate=YYYY-MM-DD&endDate=YYYY-MM-DD&page=N&size=N

        Walks all pages of the Spring Pageable response (``content`` field).

        Auth: bearer token forwarded per-request from the ``X-Snapshot-Token``
        HTTP header (bound to ContextVar by router dependency).

        Args:
            profile_id: User UUID (sent as ``userId`` query param; var name kept
                for symmetry with other internal call sites).
            start: Inclusive start date
            end: Inclusive end date
            timezone: Accepted for interface symmetry; the production pipeline
                endpoint does not consume it.
            language: Same — interface symmetry only.

        Returns:
            List of DailySnapshot models (rows that fail to parse are skipped + logged)

        Raises:
            RuntimeError: On transport failure / 4xx / 5xx
        """
        pass

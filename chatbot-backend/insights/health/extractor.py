"""Health insight data extractor — collect + prepare (no chat agent)."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from modules.data_collector import DataCollector
from modules.data_collector import TOPIC_SNAPSHOTS
from modules.data_collector import build_extract_api_fetches
from modules.data_collector import resolve_collect_topics
from modules.data_collector import strip_finance_from_raw_data
from modules.data_collector import should_collect_finance
from services.executor.constant import CalendarDataConstants
from services.executor.prepare_calendar_data import PrepareCalendarData
from services.executor.prepare_health_data import PrepareHealthData
from services.executor.prepare_time_data import PrepareTimeData
from services.external_api_service import IExternalAPIService
from utils.logger import logger

from insights.health.field_keep_maps import FIELD_KEEP_MAP, filter_by_keep_map
from insights.health.health_processor.common.constants import MONTH_LOOKBACK_DAYS
from insights.health.canonical_field_mapping import canonicalize_extracted_data

# raw_data keys produced by each FIELD_KEEP_MAP api key
_KEEP_MAP_RAW_KEYS: Dict[str, List[str]] = {
    "balance": ["balance_scores_30d", "balance_scores_7d"],
    "balance_score": ["balance_score"],
    "calendar/event": ["calendar_events", "calendar_events_week"],
    "health/summaries": ["today_health_stats"],
    "daily-user-snapshots": ["historical_snapshots"],
    "calendar/reminders": ["today_reminders"],
    "moods/latest": ["latest_mood"],
    "moods": ["moods_7d"],
    "productivity/summaries": ["productivity_summaries_7d"],
    "productivity/summary": ["productivity_summary"],
    "calendar/work-hours": ["calendar_work_hours"],
    "onboarding/users/profiles": ["user_profile"],
}


def project_health_extract_for_processor(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract normalize step: filter each API payload via FIELD_KEEP_MAP.

    For every api_key in the map, project matching raw_data fields down to
    the listed keep keys.
    Currently:
      - balance          → date / timezone / healthScore
      - calendar/event   → summary / completed / startTime / endTime / eventType
      - health/summaries → ENERGY|HR|SLEEP|STEPS blocks without id (+ SLEEP.sleepScore)
    """
    if not isinstance(raw_data, dict):
        return raw_data

    for api_key, keep_entries in FIELD_KEEP_MAP.items():
        raw_keys = _KEEP_MAP_RAW_KEYS.get(api_key) or []
        keep_keys = [
            e.get("key")
            for e in keep_entries
            if isinstance(e, dict) and e.get("key")
        ]
        for raw_key in raw_keys:
            if raw_key not in raw_data:
                continue
            before = raw_data.get(raw_key)
            cleaned = filter_by_keep_map(before, api_key)
            # List endpoints always stay lists (empty ≠ missing)
            if isinstance(before, list) and not isinstance(cleaned, list):
                cleaned = []
            raw_data[raw_key] = cleaned
        logger.info(
            f"[health_extract] projected api={api_key} keep_fields={keep_keys} "
            f"raw_keys={raw_keys}"
        )
    return raw_data


class HealthDataExtractor:
    """Extract raw + prepared health context for the health insight pipeline.

    Steps:
      1. Collect unified raw data filtered by domain/topics
         (includes TOPIC_HEALTH → today_health_stats,
          TOPIC_BALANCE → balance_score / balance_scores_7d)
      2. Extend historical snapshots to ~30d for month-window period calcs
      3. Prepare calendar / time / health params
         (PrepareHealthData → health_params via separate summaries range call)
      4. Project/normalize payloads for processor
         (health: balance rows → date / healthScore / timezone only)
    """

    def __init__(
        self,
        *,
        data_collector: DataCollector,
        external_api_service: Optional[IExternalAPIService] = None,
    ) -> None:
        self._data_collector = data_collector
        self.external_api_service = (
            external_api_service or data_collector.external_api_service
        )

    async def extract(
        self,
        user_id: str,
        timezone: Optional[str] = None,
        provider_name: str = "",
        force_update: bool = False,
        domain: Optional[str] = None,
        topics: Optional[list] = None,
    ) -> Dict[str, Any]:
        timezone = await self._data_collector.resolve_timezone(
            user_id=user_id,
            payload_timezone=timezone,
        )
        tz = ZoneInfo(timezone)
        current_dt = datetime.now(tz)
        today_date = current_dt.date()
        domain_key = (domain or "").strip().lower()
        selected_topics = resolve_collect_topics(domain=domain, topics=topics)
        selected_topics.discard(TOPIC_SNAPSHOTS)

        raw_data = await self._data_collector._collect_all_data(
            user_id=user_id,
            timezone=timezone,
            force_update=force_update,
            domain=domain,
            topics=sorted(selected_topics),
        )

        # Daily health / productivity / overall never carry finance in extract.
        if not should_collect_finance(domain=domain, topics=selected_topics):
            strip_finance_from_raw_data(raw_data)

        # Longer history for week/month period aggregates (chat-parity windows)
        extended = await self._fetch_extended_snapshots(
            user_id=user_id,
            timezone=timezone,
            today_date=today_date,
        )
        # Always keep the key after extract (empty list ≠ missing field)
        raw_data["historical_snapshots"] = extended or []

        user_profile_data = raw_data.get("user_profile") or {}
        raw_events = raw_data.get("calendar_events") or []
        prefetched_events = raw_events

        raw_data["today_reminders"] = self._filter_today_reminders(
            raw_data.get("today_reminders") or [],
            tz=tz,
            today_date=today_date,
        )

        prepare_calendar = PrepareCalendarData(
            external_api_service=self.external_api_service,
            user_id=user_id,
            provider_name=provider_name,
            timezone=timezone,
            current_dt=current_dt,
            user_profile=user_profile_data,
            prefetched_events=prefetched_events,
        )
        prepare_time = PrepareTimeData(
            user_profile=user_profile_data,
            current_dt=current_dt,
            tz=tz,
            timezone=timezone,
            calendar_events=prefetched_events or [],
        )

        calendar_data = await prepare_calendar.prepare_all_calendar_data()
        time_data = prepare_time.prepare_all_time_data()

        calendar_events = calendar_data.get(CalendarDataConstants.KEY_EVENTS, [])
        calendar_events_week = calendar_data.get(
            CalendarDataConstants.KEY_CALENDAR_EVENTS_WEEK, []
        )
        if not calendar_events_week:
            calendar_events_week = prepare_calendar.events or []
        calendar_metrics = calendar_data.get(
            CalendarDataConstants.KEY_CALENDAR_METRICS, {}
        )

        prepare_health = PrepareHealthData(
            external_api_service=self.external_api_service,
            user_id=user_id,
            user_profile=user_profile_data,
            timezone=timezone,
            time_data=time_data,
        )
        health_params = await prepare_health.prepare_all_health_params()

        # Keep both keys:
        # - today_health_stats: from collect TOPIC_HEALTH (week summaries)
        # - health_params: from PrepareHealthData (2-week range → flat dict)
        raw_data["health_params"] = health_params
        raw_data["time_data"] = time_data
        raw_data["calendar_events"] = calendar_events
        raw_data["calendar_events_week"] = calendar_events_week
        raw_data["calendar_metrics"] = calendar_metrics
        raw_data["user_profile"] = user_profile_data


        # Project upstream payloads (user_profile filter needed for all domains)
        project_health_extract_for_processor(raw_data)

        up = raw_data.get("user_profile")
        logger.info(f"[user_profile DEBUG 2] domain={domain_key} user_profile type={type(up).__name__} keys={list(up.keys()) if isinstance(up, dict) else 'N/A'} goals={up.get('goals') if isinstance(up, dict) else 'N/A'}")

        # Final guard: health/productivity/overall extract must not leak finance keys
        if not should_collect_finance(domain=domain, topics=selected_topics):
            strip_finance_from_raw_data(raw_data)

        raw_data["user_id"] = user_id
        raw_data["collect_domain"] = domain_key or raw_data.get("collect_domain")
        raw_data["collect_topics"] = sorted(selected_topics)
        raw_data["_extract_api_fetches"] = build_extract_api_fetches(
            raw_data, domain=domain_key or domain, user_id=user_id
        )

        raw_data = canonicalize_extracted_data(raw_data)

        return raw_data

    @staticmethod
    def _parse_any_datetime(raw: Any, tz: ZoneInfo) -> Optional[datetime]:
        if raw is None:
            return None
        try:
            dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=tz)
            return dt.astimezone(tz)
        except Exception:
            return None

    @staticmethod
    def _filter_today_reminders(
        reminders: List[Dict[str, Any]],
        *,
        tz: ZoneInfo,
        today_date,
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []

        for r in reminders or []:
            if not isinstance(r, dict):
                continue
            due_raw = (
                r.get("due_at")
                or r.get("dueAt")
                or r.get("dueDate")
                or r.get("due_date")
                or r.get("reminderDate")
                or r.get("startTime")
            )
            dt_utc = HealthDataExtractor._parse_any_datetime(due_raw, ZoneInfo("UTC"))
            if not dt_utc:
                logger.info(
                    f"[reminder_filter] SKIP title={r.get('title')} "
                    f"due_raw={due_raw!r} reason=unparseable"
                )
                continue
            due_date_local = dt_utc.astimezone(tz).date()
            if due_date_local == today_date:
                out.append(r)
            else:
                logger.info(
                    f"[reminder_filter] SKIP title={r.get('title')} "
                    f"due_raw={due_raw} due_local_date={due_date_local} "
                    f"today_local={today_date}"
                )
        logger.info(
            f"[reminder_filter] total_in={len(reminders)} passed={len(out)} "
            f"today_local={today_date} tz={tz}"
        )
        return out

    async def _fetch_extended_snapshots(
        self,
        user_id: str,
        timezone: str,
        today_date,
    ) -> list:
        """Fetch up to MONTH_LOOKBACK_DAYS of daily snapshots (excludes today)."""
        if not self.external_api_service:
            return []
        start_history = today_date - timedelta(days=MONTH_LOOKBACK_DAYS + 1)
        end_history = today_date - timedelta(days=1)
        try:
            snapshots = await self.external_api_service.get_daily_snapshots(
                profile_id=user_id,
                start=start_history,
                end=end_history,
                timezone=timezone,
            )
            result = [
                self._data_collector._to_dict(r) for r in (snapshots or [])
            ]
            logger.info(
                f"[health_extract] extended snapshots user={user_id} "
                f"days={MONTH_LOOKBACK_DAYS} count={len(result)}"
            )
            return result
        except Exception as e:
            logger.warning(
                f"[health_extract] extended snapshots failed for {user_id}: {e}"
            )
            return []

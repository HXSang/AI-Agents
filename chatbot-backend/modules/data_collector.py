import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Set
from zoneinfo import ZoneInfo

from agents.llm_manager import LLMManager
from clients.redis_client import RedisClient
from insights.helpers.empty_value_cleanser import clean_payload
from services.executor.cache_helpers import CacheHelpers
from services.external_api_service import IExternalAPIService
from utils.logger import logger

# Fetch topic keys for selective collection
TOPIC_PROFILE = "profile"
TOPIC_SNAPSHOTS = "snapshots"
TOPIC_CALENDAR = "calendar"
TOPIC_HEALTH = "health"
TOPIC_REMINDERS = "reminders"
TOPIC_MOOD = "mood"
TOPIC_MOOD_7D = "mood_7d"
TOPIC_PRODUCTIVITY = "productivity"
TOPIC_WORK_HOURS = "work_hours"
TOPIC_BALANCE = "balance"
TOPIC_FINANCE = "finance"

# Keys produced only when TOPIC_FINANCE is collected — never for daily health/productivity/overall.
FINANCE_RAW_KEYS: frozenset[str] = frozenset(
    {
        "finance_summary",
        "finance_goals",
        "finance_bills",
        "finance_logs",
        "finance_budgets",
        "goal_contributions",
    }
)

ALL_TOPICS: frozenset[str] = frozenset(
    {
        TOPIC_PROFILE,
        TOPIC_SNAPSHOTS,
        TOPIC_CALENDAR,
        TOPIC_HEALTH,
        TOPIC_REMINDERS,
        TOPIC_MOOD,
        TOPIC_MOOD_7D,
        TOPIC_PRODUCTIVITY,
        TOPIC_WORK_HOURS,
        TOPIC_BALANCE,
        TOPIC_FINANCE,
    }
)

# Domain → related topics (daily insight). "all" keeps legacy full fetch.
DOMAIN_TOPICS: Dict[str, frozenset[str]] = {
    "health": frozenset(
        {
            TOPIC_PROFILE,
            TOPIC_SNAPSHOTS,
            TOPIC_HEALTH,
            TOPIC_MOOD,
            TOPIC_MOOD_7D,
            TOPIC_CALENDAR,
            TOPIC_WORK_HOURS,
            TOPIC_REMINDERS,
            TOPIC_BALANCE,
        }
    ),
    "productivity": frozenset(
        {
            TOPIC_PROFILE,
            TOPIC_CALENDAR,
            TOPIC_REMINDERS,
            TOPIC_PRODUCTIVITY,
            TOPIC_WORK_HOURS,
        }
    ),
    "overall": frozenset(
        {
            TOPIC_PROFILE,
            TOPIC_SNAPSHOTS,
            TOPIC_HEALTH,
            TOPIC_MOOD,
            TOPIC_MOOD_7D,
            TOPIC_CALENDAR,
            TOPIC_REMINDERS,
            TOPIC_PRODUCTIVITY,
            TOPIC_WORK_HOURS,
            TOPIC_BALANCE,
        }
    ),
    "all": ALL_TOPICS,
}


def resolve_collect_topics(
    domain: Optional[str] = None,
    topics: Optional[Iterable[str]] = None,
) -> Set[str]:
    """Resolve which fetch topics to run.

    Priority: explicit ``topics`` > ``domain`` map > full ``all``.
    """
    if topics is not None:
        resolved = {t for t in topics if t in ALL_TOPICS}
        return resolved or set(ALL_TOPICS)
    key = (domain or "all").strip().lower()
    return set(DOMAIN_TOPICS.get(key, ALL_TOPICS))


def strip_finance_from_raw_data(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """Remove finance payload keys from a raw_data dict (in place + return)."""
    if not isinstance(raw_data, dict):
        return raw_data
    for key in FINANCE_RAW_KEYS:
        raw_data.pop(key, None)
    topics = raw_data.get("collect_topics")
    if isinstance(topics, list):
        raw_data["collect_topics"] = [t for t in topics if t != TOPIC_FINANCE]
    return raw_data


def should_collect_finance(
    domain: Optional[str] = None,
    topics: Optional[Iterable[str]] = None,
) -> bool:
    """Daily insight domains never collect finance; only explicit/all may."""
    return TOPIC_FINANCE in resolve_collect_topics(domain=domain, topics=topics)


# Demo / debug: topic → upstream API + raw_data key holding the response body.
_TOPIC_API_SPEC: Dict[str, Dict[str, str]] = {
    TOPIC_PROFILE: {
        "method": "GET",
        "path": "/api/onboarding/users/{user_id}/profiles",
        "raw_key": "user_profile",
        "label": "User profile",
    },
    TOPIC_HEALTH: {
        "method": "GET",
        "path": "/api/health/summaries",
        "raw_key": "today_health_stats",
        "label": "Health summaries (current week)",
    },
    TOPIC_MOOD: {
        "method": "GET",
        "path": "/api/moods/latest",
        "raw_key": "latest_mood",
        "label": "Latest mood",
    },
    TOPIC_MOOD_7D: {
        "method": "GET",
        "path": "/api/moods",
        "raw_key": "moods_7d",
        "label": "Moods history (7d)",
    },
    TOPIC_CALENDAR: {
        "method": "GET",
        "path": "/api/calendar/events",
        "raw_key": "calendar_events",
        "label": "Calendar events",
    },
    TOPIC_REMINDERS: {
        "method": "GET",
        "path": "/api/calendar/reminders",
        "raw_key": "today_reminders",
        "label": "Reminders",
    },
    TOPIC_WORK_HOURS: {
        "method": "GET",
        "path": "/api/calendar/work-hours",
        "raw_key": "calendar_work_hours",
        "label": "Work hours",
    },
    TOPIC_PRODUCTIVITY: {
        "method": "GET",
        "path": "/api/productivity/summaries",
        "raw_key": "productivity_summaries_7d",
        "label": "Productivity summaries (7d)",
    },
    TOPIC_BALANCE: {
        "method": "GET",
        "path": "/api/balance",
        "raw_key": "balance_scores_30d",
        "label": "Balance scores",
    },
    TOPIC_SNAPSHOTS: {
        "method": "GET",
        "path": "/api/pipeline/daily-user-snapshots",
        "raw_key": "historical_snapshots",
        "label": "Daily snapshots (short window)",
    },
}


def _jsonable_payload(value: Any) -> Any:
    """Best-effort JSON-serializable copy for demo/debug payloads."""
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json")
        except Exception:
            try:
                return value.model_dump()
            except Exception:
                return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable_payload(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_payload(v) for v in value]
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def build_extract_api_fetches(
    raw_data: Dict[str, Any],
    *,
    domain: Optional[str] = None,
    user_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List each upstream API used by HealthDataExtractor + response JSON slice.

    Collect topics come from ``raw_data.collect_topics`` / domain map. Always
    appends extractor extras: extended snapshots + prepare-health range call.
    """
    if not isinstance(raw_data, dict):
        return []

    uid = (
        user_id
        or raw_data.get("user_id")
        or (raw_data.get("user_profile") or {}).get("user_id")
        or "{user_id}"
    )
    domain_key = (
        (domain or raw_data.get("collect_domain") or "health") or "health"
    ).strip().lower()
    topics = raw_data.get("collect_topics")
    if isinstance(topics, list) and topics:
        selected = {t for t in topics if t in ALL_TOPICS}
    else:
        selected = resolve_collect_topics(domain=domain_key)
    # Extractor skips short-window snapshots in collect; uses extended fetch instead.
    selected.discard(TOPIC_SNAPSHOTS)

    fetches: List[Dict[str, Any]] = []
    seen_paths: Set[str] = set()

    def _add(
        *,
        topic: str,
        method: str,
        path: str,
        raw_key: str,
        label: str,
        source: str,
        note: str = "",
        response: Any = None,
    ) -> None:
        path_filled = path.replace("{user_id}", str(uid))
        dedupe = f"{method}:{path_filled}:{raw_key}"
        if dedupe in seen_paths:
            return
        seen_paths.add(dedupe)
        if response is None and raw_key:
            response = raw_data.get(raw_key)
        present = raw_key in raw_data if raw_key else response is not None
        if isinstance(response, list):
            count = len(response)
        elif isinstance(response, dict):
            count = len(response)
        elif response is None:
            count = 0
        else:
            count = 1
        entry: Dict[str, Any] = {
            "topic": topic,
            "label": label,
            "method": method,
            "path": path_filled,
            "raw_key": raw_key,
            "source": source,
            "present": present,
            "count": count,
            "response": _jsonable_payload(response),
        }
        if note:
            entry["note"] = note
        fetches.append(entry)

    for topic in sorted(selected):
        spec = _TOPIC_API_SPEC.get(topic)
        if not spec:
            continue
        note = ""
        if topic == TOPIC_HEALTH:
            note = (
                "Current-week summaries → today_health_stats; "
                "field_keep_map health/summaries keeps type/dateRange/"
                "daysWithData/createdAt/updatedAt/summaryData/data "
                "(+ SLEEP.sleepScore); drops id"
            )
        elif topic == TOPIC_BALANCE and domain_key == "health":
            note = (
                "Balance range → balance_score + balance_scores_30d; "
                "field_keep_map keeps date / timezone / healthScore"
            )
        elif topic == TOPIC_CALENDAR and domain_key == "health":
            note = (
                "field_keep_map keeps summary / completed / startTime / "
                "endTime / eventType"
            )
        _add(
            topic=topic,
            method=spec["method"],
            path=spec["path"],
            raw_key=spec["raw_key"],
            label=spec["label"],
            source="collect",
            note=note,
        )
        # Balance also surfaces today's row separately from the 7d list
        if topic == TOPIC_BALANCE and "balance_score" in raw_data:
            _add(
                topic=topic,
                method=spec["method"],
                path=spec["path"],
                raw_key="balance_score",
                label="Balance score (today)",
                source="collect",
                note=(
                    "Today slice from /api/balance; "
                    "field_keep_map keeps date / timezone / healthScore"
                ),
            )
        # Productivity also surfaces today's slice
        if topic == TOPIC_PRODUCTIVITY and "productivity_summary" in raw_data:
            _add(
                topic=topic,
                method=spec["method"],
                path=spec["path"],
                raw_key="productivity_summary",
                label="Productivity summary (today slice)",
                source="collect",
                note="Derived from productivity_summaries_7d for today",
            )

    # Always fetched by HealthDataExtractor (replaces short TOPIC_SNAPSHOTS collect)
    _add(
        topic="snapshots_extended",
        method="GET",
        path="/api/pipeline/daily-user-snapshots",
        raw_key="historical_snapshots",
        label="Daily snapshots (~30d)",
        source="health_extractor",
        note="Extended lookback for period aggregates (replaces short collect snapshots)",
    )

    # Separate prepare call: 2-week range → flat health_params
    _add(
        topic="health_prepare",
        method="GET",
        path="/api/health/summaries",
        raw_key="health_params",
        label="Health summaries (2-week range → health_params)",
        source="prepare_health",
        note=(
            "PrepareHealthData.get_health_summaries_by_range; "
            "response shown is derived health_params (not raw summaries list)"
        ),
        response=raw_data.get("health_params"),
    )

    return fetches


class DataCollector:
    def __init__(self):
        self.external_api_service: Optional[IExternalAPIService] = None
        self.redis_client = RedisClient()

        logger.info("✅ DataCollector initialized")

    # Last-resort fallback. Comes from env ``DEFAULT_TIMEZONE`` so deploys
    # in other regions aren't silently locked to VN. UTC if unset — callers
    # downstream rely on resolver output and UTC here just means "no source
    # provided any TZ at all". Use ``Asia/Ho_Chi_Minh`` for the legacy
    # VN-only behaviour.
    DEFAULT_TIMEZONE = os.getenv("DEFAULT_TIMEZONE", "UTC")

    async def resolve_timezone(
        self,
        user_id: Optional[str],
        payload_timezone: Optional[str],
    ) -> str:
        if payload_timezone and payload_timezone.strip():
            return payload_timezone.strip()
        if user_id:
            profile = await self._fetch_user_profile(user_id)
            if isinstance(profile, dict):
                tz = (
                    profile.get("timezone")
                    or profile.get("time_data_timezone")
                )
                if tz and isinstance(tz, str) and tz.strip():
                    return tz.strip()
        return self.DEFAULT_TIMEZONE

    async def _fetch_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        try:
            general_data = await CacheHelpers.fetch_general_user_data_cached(
                self.redis_client,
                self.external_api_service,
                user_id,
                log_prefix="Unified Data Collection",
            )
            if general_data:
                return general_data
            logger.warning(
                f"⚠️ [Unified Data Collection] user_profile empty for "
                f"{user_id}: upstream returned no profile. This usually means "
                f"the user hasn't completed onboarding OR the profile is in a "
                f"different DB than this service can read. Downstream narrative "
                f"will fall back to a generic (no-identity) bg."
            )
            return {"_missing": True, "reason": "upstream_returned_none"}
        except Exception as e:
            logger.warning(
                f"⚠️ Failed to fetch user profile for {user_id}: {str(e)}. "
                f"Narrative will run without identity fields."
            )
            return {"_missing": True, "reason": f"exception:{type(e).__name__}"}

    async def _fetch_in_chunks(
        self,
        fetch_func,
        user_id: str,
        timezone: str,
        start_date: datetime.date,
        end_date: datetime.date,
        chunk_days: int = 30,
    ) -> List[Any]:
        """Fetch data in chunks to avoid API limits (e.g., max 31 days per request)."""
        tasks = []
        current_start = start_date
        while current_start <= end_date:
            current_end = min(current_start + timedelta(days=chunk_days - 1), end_date)
            tasks.append(
                fetch_func(
                    user_id=user_id,
                    start_date=current_start.strftime("%Y-%m-%d"),
                    end_date=current_end.strftime("%Y-%m-%d"),
                    timezone=timezone,
                )
            )
            current_start = current_end + timedelta(days=1)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_data = []
        for res in results:
            if isinstance(res, list):
                all_data.extend(res)
            elif isinstance(res, Exception):
                logger.error(f"Error in chunked fetch: {str(res)}")
        return all_data

    _EXPECTED_SNAPSHOT_FIELDS = (
        "h_sleep_hours",
        "h_bedtime",
        "h_wake_time",
        "h_hrv_score",
        "h_sleep_quality",
        "m_primary_mood",
        "m_mood_score_avg",
    )

    @staticmethod
    def _to_dict(row: Any) -> Dict[str, Any]:
        """Convert SQLAlchemy model or dict to plain dict for serialization."""
        if isinstance(row, dict):
            return row
        if hasattr(row, "__dict__"):
            return {k: v for k, v in row.__dict__.items() if not k.startswith("_")}
        return {}

    @staticmethod
    def _get_today_from_summaries(
        summaries: Any, today_date: datetime.date
    ) -> Optional[Any]:
        if not summaries:
            return None
        today_str = today_date.strftime("%Y-%m-%d")
        last_dict: Optional[Any] = None
        for item in summaries:
            if isinstance(item, dict):
                last_dict = item
                item_date = (
                    item.get("date")
                    or item.get("dateRange", {}).get("startDate", "")
                )
                if item_date == today_str:
                    return item
            elif hasattr(item, "dateRange"):
                last_dict = item
                item_date = (
                    item.dateRange.startDate
                    if hasattr(item.dateRange, "startDate")
                    else getattr(item.dateRange, "startDate", "")
                )
                if item_date == today_str:
                    return item
        # Fallback: upstream didn't return a date field — assume last row = today
        if last_dict is not None and isinstance(last_dict, dict) and "date" not in last_dict:
            last_dict["date"] = today_str
        return last_dict

    async def _fetch_historical_snapshots(
        self, user_id: str, timezone: str, today_date: datetime.date
    ) -> Optional[List[Dict[str, Any]]]:
        start_history = today_date - timedelta(days=8)
        end_history = today_date - timedelta(days=1)
        try:
            snapshots = await self.external_api_service.get_daily_snapshots(
                profile_id=user_id,
                start=start_history,
                end=end_history,
                timezone=timezone,
            )
            result = [self._to_dict(r) for r in (snapshots or [])]
            if not result:
                return None
            sample = result[0]
            missing = [
                f for f in self._EXPECTED_SNAPSHOT_FIELDS if sample.get(f) is None
            ]
            if missing:
                logger.info(
                    f"ℹ️ Daily snapshots for {user_id} missing columns {missing} "
                    f"on sample row — sleep/mood trend signals will be degraded"
                )
            return result
        except Exception as e:
            logger.error(f"Error fetching historical snapshots: {str(e)}")
            return None

    async def _fetch_calendar_events(
        self, user_id: str, timezone: str, today_date: datetime.date
    ) -> List[Any]:
        monday = today_date - timedelta(days=today_date.weekday())
        start_today_dt = monday.strftime("%Y-%m-%dT00:00")
        end_calendar_dt = (today_date + timedelta(days=7)).strftime("%Y-%m-%dT23:59")
        try:
            return await self.external_api_service.get_calendar_events(
                user_id=user_id,
                start_date_time=start_today_dt,
                end_date_time=end_calendar_dt,
                timezone=timezone,
            )
        except Exception as e:
            logger.error(f"Error fetching calendar events: {str(e)}")
            return []

    _EXPECTED_HEALTH_FIELDS = (
        "hrv_score",
        "sleep_quality_score",
        "bedtime",
        "first_sleep_time",
        "wake_time",
        "resting_hr",
    )

    async def _fetch_health_summaries(
        self, user_id: str, timezone: str
    ) -> Dict[str, Any]:
        try:
            data = await self.external_api_service.get_health_summaries(
                user_id=user_id, timezone=timezone
            )
            data = data or {}
            def _serialize(v: Any) -> Any:
                if hasattr(v, "model_dump"):
                    return v.model_dump(mode="json")
                if hasattr(v, "__dict__"):
                    return {k: _serialize(val) for k, val in v.__dict__.items()}
                return v

            data = {k: _serialize(v) for k, v in data.items()}
            missing = [f for f in self._EXPECTED_HEALTH_FIELDS if data.get(f) is None]
            if missing:
                logger.info(
                    f"ℹ️ Health summary for {user_id} missing fields {missing} "
                    f"— DataProcessor will attempt to backfill from historical_snapshots"
                )
            return data
        except Exception as e:
            logger.error(f"Error fetching health summaries: {str(e)}")
            return {}

    async def _fetch_reminders(
        self, user_id: str, timezone: str, today_date: datetime.date
    ) -> List[Any]:
        start_today_date = today_date.strftime("%Y-%m-%d")
        end_date = (today_date + timedelta(days=6)).strftime("%Y-%m-%d")
        try:
            return await self.external_api_service.get_reminders(
                user_id=user_id,
                start_date=start_today_date,
                end_date=end_date,
                timezone=timezone,
            )
        except Exception as e:
            logger.error(f"Error fetching reminders: {str(e)}")
            return []

    async def _fetch_mood(
        self, user_id: str, timezone: str, today_date: datetime.date
    ) -> Any:
        try:
            return await self.external_api_service.get_latest_mood(
                user_id=user_id,
                target_date=today_date.isoformat(),
                timezone=timezone,
            )
        except Exception as e:
            logger.error(f"Error fetching mood: {str(e)}")
            return None

    @staticmethod
    def _normalize_latest_mood(
        mood: Optional[Dict[str, Any]],
        tz: ZoneInfo,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(mood, dict):
            return mood
        raw_date = (
            mood.get("mood_date")
            or mood.get("current_mood_date")
            or mood.get("date")
        )
        if not raw_date:
            return mood
        try:
            dt = datetime.fromisoformat(str(raw_date).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                # Backend trả naive datetime: mặc định coi như UTC
                dt = dt.replace(tzinfo=ZoneInfo("UTC"))
            local_iso = dt.astimezone(tz).date().isoformat()
            # Copy shallow để tránh mutate dict nguồn
            return {**mood, "mood_date_local": local_iso}
        except Exception:
            # Không phá vỡ pipeline nếu parse lỗi
            return mood

    async def _fetch_moods_7d(
        self, user_id: str, timezone: str, today_date: datetime.date
    ) -> Optional[List[Any]]:
        """Fetch last-7-days mood history.

        Returns ``None`` (NOT ``[]``) when the API returns no entries so the
        collector can drop the ``moods_7d`` key from the payload.  An empty
        list here would feed ``data_processor._count_mood_days``,
        ``MoodSignalProcessor._build_history`` and the
        ``behavioral_pattern_processor`` loop with nothing to iterate over —
        wasted work and a confusing "moods_7d": [] block in the debug HTML.
        Downstream uses ``raw_data.get("moods_7d") or []`` which collapses
        None to ``[]`` automatically.
        """
        tzinfo = ZoneInfo(timezone)
        try:
            start_dt = datetime.combine(
                today_date - timedelta(days=7),
                datetime.min.time(),
                tzinfo=tzinfo,
            )
            end_dt = datetime.combine(
                today_date - timedelta(days=1),
                datetime.max.time(),
                tzinfo=tzinfo,
            )
            resp = await self.external_api_service.get_moods(
                user_id=user_id,
                page=1,
                size=50,
                start_datetime=start_dt.isoformat(),
                end_datetime=end_dt.isoformat(),
                timezone=timezone,
            )
            if isinstance(resp, dict):
                content = resp.get("content")
                if isinstance(content, list):
                    return content if content else None
                return None
            if isinstance(resp, list):
                return resp if resp else None
            return None
        except Exception as e:
            logger.error(f"Error fetching moods 7d: {str(e)}")
            return None

    async def _fetch_productivity_summaries_7d(
        self, user_id: str, timezone: str, today_date: datetime.date
    ) -> Optional[List[Any]]:
        """Fetch last-7-days productivity summaries.

        Returns ``None`` (NOT ``[]``) when the API returns no entries so the
        collector can drop the ``productivity_summaries_7d`` key from the
        payload.  An empty list here would feed productivity processors and
        the debug HTML with nothing useful.  Downstream consumers use
        ``raw_data.get("productivity_summaries_7d") or []`` so a missing
        key is harmless.
        """
        try:
            start = (today_date - timedelta(days=7)).strftime("%Y-%m-%d")
            end = today_date.strftime("%Y-%m-%d")
            raw = (
                await self.external_api_service.get_productivity_summaries_by_range(
                    user_id=user_id,
                    start_date=start,
                    end_date=end,
                    timezone=timezone,
                )
                or []
            )
            if not raw:
                return None
            window_start = today_date - timedelta(days=7)
            enriched: List[Any] = []
            for i, item in enumerate(raw):
                if not isinstance(item, dict):
                    enriched.append(item)
                    continue
                if not item.get("date"):
                    date_range = item.get("dateRange") or item.get("date_range")
                    start_str = None
                    if isinstance(date_range, dict):
                        start_str = date_range.get("startDate") or date_range.get(
                            "start_date"
                        )
                    if start_str:
                        item["date"] = str(start_str)[:10]
                    elif "dateRange" not in item and "date_range" not in item:
                        item["date"] = (window_start + timedelta(days=i)).strftime(
                            "%Y-%m-%d"
                        )
                enriched.append(item)
            return enriched if enriched else None
        except Exception as e:
            logger.error(f"Error fetching productivity summaries 7d: {str(e)}")
            return None

    async def _fetch_work_hours(
        self, user_id: str, timezone: str, today_date: datetime.date
    ) -> Any:
        """Fetch work hours for a specific date."""
        try:
            date_str = today_date.strftime("%Y-%m-%d")
            return await self.external_api_service.get_work_hours(
                user_id=user_id,
                date=date_str,
                timezone=timezone,
            )
        except Exception as e:
            logger.error(f"Error fetching work-hours: {str(e)}")
            return None

    async def _fetch_balance_score(
        self, user_id: str, timezone: str, today_date: datetime.date
    ) -> Dict[str, Any]:
        try:
            start_date = (today_date - timedelta(days=30)).strftime("%Y-%m-%d")
            end_date = today_date.strftime("%Y-%m-%d")
            today_str = today_date.strftime("%Y-%m-%d")

            scores: List[Any] = []
            try:
                raw = await self.external_api_service.get_balance_scores_by_range(
                    user_id=user_id,
                    start_date=start_date,
                    end_date=end_date,
                    timezone=timezone,
                )
                if isinstance(raw, list):
                    scores = sorted(
                        [
                            s
                            for s in raw
                            if isinstance(s, dict)
                            and any(
                                s.get(k) is not None
                                for k in (
                                    "balanceScore",
                                    "balance_health_score",
                                    "healthScore",
                                    "health_score",
                                )
                            )
                        ],
                        key=lambda x: x.get("date") or "",
                    )
            except Exception as inner_e:
                logger.warning(f"balance_scores fetch failed: {inner_e}")

            # Extract today's score and past 30d (days < today)
            today_score: Optional[Any] = None
            past_30d: List[Any] = []

            for item in scores:
                item_date = item.get("date") if isinstance(item, dict) else getattr(item, "date", "")
                if item_date == today_str:
                    today_score = item
                elif item_date < today_str:
                    past_30d.append(item)

            return {"today": today_score, "past_30d": past_30d}
        except Exception as e:
            logger.error(f"Error fetching balance score: {str(e)}")
            return {"today": None, "past_7d": []}

    async def _fetch_finance_summary(
        self, user_id: str, today_date: datetime.date
    ) -> Any:
        try:
            # Requires month format "YYYY-MM"
            return await self.external_api_service.get_finance_summary(
                user_id=user_id, month=today_date.strftime("%Y-%m")
            )
        except Exception as e:
            logger.error(f"Error fetching finance summary: {str(e)}")
            return None

    async def _fetch_finance_goals(self, user_id: str) -> Any:
        try:
            return await self.external_api_service.get_finance_goals(user_id=user_id)
        except Exception as e:
            logger.error(f"Error fetching finance goals: {str(e)}")
            return []

    async def _fetch_finance_bills(self, user_id: str) -> Any:
        try:
            return await self.external_api_service.get_finance_bills(user_id=user_id)
        except Exception as e:
            logger.error(f"Error fetching finance bills: {str(e)}")
            return []

    async def _fetch_finance_logs(
        self, user_id: str, month: Optional[str] = None
    ) -> Any:
        """Fetch finance logs (detailed transactions) for current month."""
        try:
            return await self.external_api_service.get_finance_logs(
                user_id=user_id,
                month=month,
                size=100,  # Get up to 100 recent transactions
            )
        except Exception as e:
            logger.error(f"Error fetching finance logs: {str(e)}")
            return []

    async def _fetch_finance_budgets(
        self,
        user_id: str,
        budget_type: Optional[str] = None,
        active: bool = True,
    ) -> Any:
        """Fetch finance budgets (budget vs actual tracking)."""
        try:
            return await self.external_api_service.get_finance_budgets(
                user_id=user_id,
                budget_type=budget_type,
                active=active,
            )
        except Exception as e:
            logger.error(f"Error fetching finance budgets: {str(e)}")
            return []

    @staticmethod
    def _build_goal_contributions(finance_goals: Any) -> List[Dict[str, Any]]:
        try:
            if not isinstance(finance_goals, list):
                return []
            goal_contributions = []
            for goal in finance_goals:
                if not isinstance(goal, dict):
                    continue
                goal_contributions.append({
                    "goalName": goal.get("name", ""),
                    "category": goal.get("category", goal.get("goalType", "")).upper(),
                    "targetAmount": goal.get("targetAmount", 0),
                    "contributed": goal.get("contributionAmount", 0),
                    "currentAmount": goal.get("currentAmount", 0),
                    "targetDate": goal.get("targetDate", ""),
                    "progressPercentage": goal.get("progressPercentage", 0),
                })
            return goal_contributions
        except Exception as e:
            logger.warning(f"⚠️ Failed to build goal_contributions: {e}")
            return []

    async def _collect_all_data(
        self,
        user_id: str,
        timezone: Optional[str] = None,
        force_update: bool = False,
        domain: Optional[str] = None,
        topics: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        """Fetch BE data in parallel, optionally filtered by domain/topics.

        Args:
            domain: ``health`` | ``productivity`` | ``overall`` | ``all``
            topics: explicit topic set (overrides domain when provided)
        """
        # Centralized timezone resolution: payload → profile → DEFAULT_TIMEZONE.
        timezone = await self.resolve_timezone(user_id, timezone)
        tz = ZoneInfo(timezone)
        now = datetime.now(tz)
        today_date = now.date()
        selected = resolve_collect_topics(domain=domain, topics=topics)
        current_month = today_date.strftime("%Y-%m")

        logger.info(
            f"🔄 Fetching raw data for user {user_id} "
            f"domain={domain or 'all'} topics={sorted(selected)}"
        )

        jobs: Dict[str, Any] = {}
        if TOPIC_PROFILE in selected:
            jobs[TOPIC_PROFILE] = self._fetch_user_profile(user_id)
        if TOPIC_SNAPSHOTS in selected:
            jobs[TOPIC_SNAPSHOTS] = self._fetch_historical_snapshots(
                user_id, timezone, today_date
            )
        if TOPIC_CALENDAR in selected:
            jobs[TOPIC_CALENDAR] = self._fetch_calendar_events(
                user_id, timezone, today_date
            )
        if TOPIC_HEALTH in selected:
            jobs[TOPIC_HEALTH] = self._fetch_health_summaries(user_id, timezone)
        if TOPIC_REMINDERS in selected:
            jobs[TOPIC_REMINDERS] = self._fetch_reminders(
                user_id, timezone, today_date
            )
        if TOPIC_MOOD in selected:
            jobs[TOPIC_MOOD] = self._fetch_mood(user_id, timezone, today_date)
        if TOPIC_MOOD_7D in selected:
            jobs[TOPIC_MOOD_7D] = self._fetch_moods_7d(
                user_id, timezone, today_date
            )
        if TOPIC_PRODUCTIVITY in selected:
            jobs[TOPIC_PRODUCTIVITY] = self._fetch_productivity_summaries_7d(
                user_id, timezone, today_date
            )
        if TOPIC_WORK_HOURS in selected:
            jobs[TOPIC_WORK_HOURS] = self._fetch_work_hours(
                user_id, timezone, today_date
            )
        if TOPIC_BALANCE in selected:
            jobs[TOPIC_BALANCE] = self._fetch_balance_score(
                user_id, timezone, today_date
            )
        if TOPIC_FINANCE in selected:
            jobs["finance_summary"] = self._fetch_finance_summary(
                user_id, today_date
            )
            jobs["finance_goals"] = self._fetch_finance_goals(user_id)
            jobs["finance_bills"] = self._fetch_finance_bills(user_id)
            jobs["finance_logs"] = self._fetch_finance_logs(
                user_id, month=current_month
            )
            jobs["finance_budgets"] = self._fetch_finance_budgets(user_id)

        fetched: Dict[str, Any] = {}
        if jobs:
            keys = list(jobs.keys())
            values = await asyncio.gather(
                *[jobs[k] for k in keys], return_exceptions=True
            )
            fetched = dict(zip(keys, values))

        def _get(key: str, default: Any) -> Any:
            if key not in fetched:
                return default
            val = fetched[key]
            return default if isinstance(val, Exception) else val

        prod_7d = _get(TOPIC_PRODUCTIVITY, None)
        if not isinstance(prod_7d, list):
            prod_7d = None
        balance = _get(TOPIC_BALANCE, {"today": None, "past_7d": []})
        if not isinstance(balance, dict):
            balance = {"today": None, "past_7d": []}

        finance_goals = _get("finance_goals", [])
        result = {
            "user_profile": _get(TOPIC_PROFILE, None),
            "calendar_events": _get(TOPIC_CALENDAR, []),
            "today_health_stats": _get(TOPIC_HEALTH, {}),
            "today_reminders": _get(TOPIC_REMINDERS, []),
            "latest_mood": DataCollector._normalize_latest_mood(
                _get(TOPIC_MOOD, None), tz
            ),
            "productivity_summary": DataCollector._get_today_from_summaries(
                prod_7d, today_date
            ),
            "calendar_work_hours": _get(TOPIC_WORK_HOURS, []) or [],
            "balance_score": balance.get("today"),
            "balance_scores_30d": balance.get("past_30d") or [],
            "timezone": timezone,
            "current_time": now.isoformat(),
            "collect_domain": domain or "all",
            "collect_topics": sorted(selected),
        }
        # Only include productivity_summaries_7d when we actually have entries.
        # The fetcher returns None for "API responded but empty / failed" so
        # we don't pollute the payload with a useless "productivity_summaries_7d": []
        # block.  Downstream consumers use ``raw_data.get(...) or []`` so a
        # missing key is harmless.
        if prod_7d:
            result["productivity_summaries_7d"] = prod_7d
        # Only include moods_7d when we actually have entries.
        moods_7d = _get(TOPIC_MOOD_7D, None)
        if moods_7d:
            result["moods_7d"] = moods_7d
        # Only include historical_snapshots when we actually have rows.
        snapshots = _get(TOPIC_SNAPSHOTS, None)
        if snapshots:
            result["historical_snapshots"] = snapshots
        if TOPIC_FINANCE in selected:
            result["finance_summary"] = _get("finance_summary", None)
            result["finance_goals"] = (
                finance_goals if isinstance(finance_goals, list) else []
            )
            result["finance_bills"] = _get("finance_bills", [])
            result["finance_logs"] = _get("finance_logs", [])
            result["finance_budgets"] = _get("finance_budgets", [])
            result["goal_contributions"] = DataCollector._build_goal_contributions(
                finance_goals
            )

        # Keep empty top-level lists (e.g. balance_scores_30d=[]) so downstream
        # code can distinguish "API returned empty" (key present, list=[]) from
        # "API not called" (key absent).  Only strip scalar null/empty values
        # and empty dict items *inside* lists.  preserve_empty_items=True keeps
        # {} items in lists so downstream can see "this row exists but has no data".
        cleaned = clean_payload(
            result,
            preserve_empty_items=True,
            log_tag=f"collect:{domain or 'all'}",
        )

        for k in (
            "user_profile",
            "timezone",
            "current_time",
            "collect_domain",
            "collect_topics",
            "user_id",
        ):
            if k in result and k not in cleaned:
                cleaned[k] = result[k]
        return cleaned

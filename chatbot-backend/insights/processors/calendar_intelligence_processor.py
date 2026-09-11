import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import dateutil.parser

logger = logging.getLogger(__name__)


from insights.insight_config import CalendarConfig
from insights.processors.reminder_signal_processor import extract_reminder_signals
from insights.schemas.processed_context import CalendarEventBrief
from insights.schemas.processed_context import CalendarIntelligenceBlock
from insights.schemas.processed_context import CurrentEvent
from insights.schemas.processed_context import FreeWindow
from insights.schemas.processed_context import MeetingTypeBreakdown
from insights.schemas.processed_context import NextEvent
from insights.schemas.processed_context import PreviousEvent
from insights.schemas.processed_context import UpcomingReminder
from services.executor.constant import CalendarEventCategoryConstants
from services.executor.constant import HealthDataConstants
from services.executor.constant import TimeDataConstants


_EVENT_START_KEYS = (
    "startTime",
    "start_time",
    "calendar_event_start_time",
    "calendar_event_startDateTime",
)
_EVENT_END_KEYS = (
    "endTime",
    "end_time",
    "calendar_event_end_time",
    "calendar_event_endDateTime",
)

# Raw calendar eventType values that belong in the MEETINGS category.
# CalendarEventCategoryConstants.ALLOWED has MEETINGS (plural) but not
# MEETING / CONFERENCE, so resolve() would otherwise map them to OTHER.
_EVENT_TYPE_TO_CATEGORY = {
    "MEETING": "MEETINGS",
    "MEETINGS": "MEETINGS",
    "CONFERENCE": "MEETINGS",
}


def _event_time_str(
    event: Dict[str, Any], nested_key: str, keys: tuple[str, ...]
) -> str:
    """Return the first usable time string from an event.

    ``start``/``end`` may be dicts (``{"dateTime": ...}``) or plain ISO
    strings depending on the upstream source; handle both so a bare-string
    value never raises AttributeError.
    """
    nested = event.get(nested_key)
    if isinstance(nested, dict):
        val = nested.get("dateTime") or nested.get("date")
        if val:
            return str(val)
    elif isinstance(nested, str) and nested.strip():
        return nested
    for k in keys:
        val = event.get(k)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def _event_start_str(event: Dict[str, Any]) -> str:
    return _event_time_str(event, "start", _EVENT_START_KEYS)


def _event_end_str(event: Dict[str, Any]) -> str:
    return _event_time_str(event, "end", _EVENT_END_KEYS)


def _ensure_comparable(dt: datetime, ref: datetime) -> datetime:
    """Normalize ``dt`` to ``ref``'s awareness so ``dt`` and ``ref`` compare.

    Mixed tz-aware / naive comparisons raise ``TypeError``; the callers
    swallow it and silently drop the event, which is what emptied
    ``next_event``/``previous_event``/``upcoming_events`` and made
    ``free_windows`` fall back to ``now``→bedtime. Handles both directions
    plus aware→aware across differing tzinfo.
    """
    if dt.tzinfo is None and ref.tzinfo is not None:
        return dt.replace(tzinfo=ref.tzinfo)
    if dt.tzinfo is not None and ref.tzinfo is None:
        # Compare against wall-clock time in the event's own zone.
        return dt.replace(tzinfo=None)
    if dt.tzinfo is not None and ref.tzinfo is not None:
        return dt.astimezone(ref.tzinfo)
    return dt


def _event_title(event: Dict[str, Any]) -> str:
    return (
        event.get("summary")
        or event.get("title")
        or event.get("calendar_event_title")
        or "Untitled"
    )


def _event_category(event: Dict[str, Any]) -> Optional[str]:
    raw = event.get("category")
    raw_present = raw is not None and str(raw).strip() != ""
    if raw_present:
        resolved = CalendarEventCategoryConstants.resolve(raw)
        if resolved != "OTHER":
            return resolved

    # Category missing or unmapped → fall back to eventType so raw
    # MEETING/CONFERENCE types don't collapse to OTHER.
    event_type = (
        event.get("eventType")
        or event.get("event_type")
        or event.get("calendar_event_type")
    )
    if event_type is not None and str(event_type).strip() != "":
        mapped = _EVENT_TYPE_TO_CATEGORY.get(str(event_type).strip().upper())
        if mapped:
            return CalendarEventCategoryConstants.resolve(mapped)

    if raw_present:
        return CalendarEventCategoryConstants.resolve(raw)
    return None


def _event_type(event: Dict[str, Any]) -> Optional[str]:
    raw = (
        event.get("eventType")
        or event.get("event_type")
        or event.get("calendar_event_type")
    )
    if raw is None or str(raw).strip() == "":
        return None
    return str(raw).strip().lower()


def _label_for_hour(hour: int) -> str:
    if 4 <= hour < 12:
        return "open_morning"
    if 12 <= hour < 17:
        return "open_afternoon"
    if 17 <= hour < 22:
        return "open_evening"
    return "open_night"


class CalendarIntelligenceProcessor:
    """
    Block 3: Calendar Intelligence Processor
    Computes cognitive load, parses free windows (ignoring < 10 mins),
    and categorizes meetings.
    """

    @staticmethod
    def process(
        calendar_events: List[Dict[str, Any]],
        calendar_metrics: Dict[str, Any],
        time_data: Dict[str, Any],
        reminders: Optional[List[Dict[str, Any]]] = None,
        user_timezone: Optional[str] = None,
    ) -> CalendarIntelligenceBlock:
        current_time_str = time_data.get(TimeDataConstants.KEY_CURRENT_TIME_ISO)

        tz_value = (
            user_timezone
            or (time_data or {}).get(TimeDataConstants.KEY_TIMEZONE)
            or (time_data or {}).get(TimeDataConstants.KEY_TZ)
        )

        if not current_time_str:
            if tz_value:
                try:
                    now = datetime.now(ZoneInfo(tz_value) if isinstance(tz_value, str) else tz_value)
                except Exception:
                    logger.warning(
                        "CalendarIntelligenceProcessor: invalid tz %r, falling back to naive now()",
                        tz_value,
                    )
                    now = datetime.now()
            else:
                logger.warning(
                    "CalendarIntelligenceProcessor: 'current_time_iso' AND tz both missing — "
                    "using naive now(). Caller MUST forward user timezone."
                )
                now = datetime.now()
        else:
            now = dateutil.parser.isoparse(current_time_str)
            if tz_value:
                try:
                    tz_obj = (
                        ZoneInfo(tz_value)
                        if isinstance(tz_value, str)
                        else tz_value
                    )
                    if now.tzinfo is None:
                        now = now.replace(tzinfo=tz_obj)
                    else:
                        now = now.astimezone(tz_obj)
                except Exception as ex:
                    logger.warning(
                        "CalendarIntelligenceProcessor: failed to apply tz %r: %s",
                        tz_value, ex,
                    )
            elif now.tzinfo is None:
                logger.error(
                    "CalendarIntelligenceProcessor: naive current_time_iso=%r without user tz — "
                    "refusing to fabricate UTC. Pass user_timezone.",
                    current_time_str,
                )

        logger.info(
            "[cal_intel] now=%s | tz_value=%r | current_time_iso=%r | "
            "time_data.timezone=%r | time_data.tz=%r | user_timezone=%r",
            now.isoformat(),
            tz_value,
            current_time_str,
            (time_data or {}).get(TimeDataConstants.KEY_TIMEZONE),
            (time_data or {}).get(TimeDataConstants.KEY_TZ),
            user_timezone,
        )

        # Use user-local date, not UTC date (avoids off-by-7h for +07:00 users)
        current_date_str = now.date().isoformat()

        # 1. Cognitive Load
        b2b_count, continuous_minutes = (
            CalendarIntelligenceProcessor._compute_meeting_density(
                calendar_events, current_date_str, now
            )
        )

        b2b_t = CalendarConfig.B2B_COUNT_THRESHOLDS
        cont_t = CalendarConfig.CONTINUOUS_MINUTES_THRESHOLDS

        if b2b_count >= b2b_t[0] or continuous_minutes >= cont_t[0]:
            cognitive_load = "overloaded"
        elif b2b_count >= b2b_t[1] or continuous_minutes >= cont_t[1]:
            cognitive_load = "high"
        elif b2b_count >= b2b_t[2] or continuous_minutes >= cont_t[2]:
            cognitive_load = "moderate"
        else:
            cognitive_load = "low"

        # (Moved up to support _compute_meeting_density)

        # 2. Free Windows
        free_windows = CalendarIntelligenceProcessor._compute_free_windows(
            calendar_events, now, time_data
        )
        logger.info(f"[Debug] free_windows_today count: {len(free_windows)}")
        logger.info(f"[Debug] free_windows_today: {free_windows}")

        # 3. Event States
        next_event, current_event, previous_event = (
            CalendarIntelligenceProcessor._compute_event_states(calendar_events, now)
        )
        upcoming_events = CalendarIntelligenceProcessor._compute_upcoming_briefs(
            calendar_events, now, limit=6
        )

        # 4. Meeting Breakdown
        breakdown = CalendarIntelligenceProcessor._compute_meeting_breakdown(
            calendar_events, current_date_str
        )

        # 5. Recent Meeting Count (past 2 hours)
        recent_count = CalendarIntelligenceProcessor._compute_recent_meeting_count(
            calendar_events, now
        )

        # 6. Reminder-derived signals (next reminder + overdue titles).
        #    These flow into the "current state" narrative so the LLM can
        #    surface concrete reminders, not just counts.
        nearest_reminder_obj: Optional[UpcomingReminder] = None
        overdue_titles: List[str] = []
        try:
            tz_name = (
                user_timezone
                or time_data.get("timezone")
                or "UTC"
            )
            try:
                user_tz = ZoneInfo(tz_name)
            except Exception:
                user_tz = ZoneInfo("UTC")
            # Ensure `now` is timezone-aware (the upstream parsing already
            # gives us an aware datetime, but guard anyway).
            now_for_compare = (
                now if now.tzinfo is not None else now.replace(tzinfo=user_tz)
            )
            reminder_signals = extract_reminder_signals(
                raw_reminders=reminders or [],
                raw_calendar_events=calendar_events or [],
                now_utc=now_for_compare.astimezone(timezone.utc),
                user_tz=user_tz,
            )
            overdue_titles = list(
                reminder_signals.get("overdue_reminder_titles") or []
            )
            nr = reminder_signals.get("nearest_reminder")
            if isinstance(nr, dict):
                nearest_reminder_obj = UpcomingReminder(**nr)
        except Exception as e:
            logger.warning(
                "CalendarIntelligenceProcessor: reminder signals failed: %s", e
            )

        return CalendarIntelligenceBlock(
            cognitive_load_so_far=cognitive_load,
            back_to_back_count=b2b_count,
            continuous_minutes=continuous_minutes,
            free_windows_today=free_windows,
            next_event=next_event,
            current_event=current_event,
            previous_event=previous_event,
            upcoming_events=upcoming_events,
            meeting_type_breakdown=breakdown,
            meeting_minutes_today=(
                CalendarIntelligenceProcessor._safe_get_float(
                    calendar_metrics, "meeting_minutes"
                )
                or 0.0
            ),
            recent_meeting_count_2h=recent_count,
            nearest_reminder=nearest_reminder_obj,
            overdue_reminder_titles=overdue_titles,
            # Enrichment — surface metrics already produced by
            # ``prepare_calendar_data`` and the BE itself, so the LLM
            # can quote them verbatim without re-deriving.
            work_events_hours_week=CalendarIntelligenceProcessor._safe_get_float(
                calendar_metrics, "work_events_hours_week"
            ),
            work_events_hours_weekend=CalendarIntelligenceProcessor._safe_get_float(
                calendar_metrics, "work_events_hours_weekend"
            ),
            work_load_high=CalendarIntelligenceProcessor._safe_get_bool(
                calendar_metrics, "work_load_high"
            ),
            calendar_density=CalendarIntelligenceProcessor._safe_get_float(
                calendar_metrics, "calendar_density"
            ),
            # Today's event aggregates — computed here from
            # ``calendar_events`` because the BE only ships per-day metrics.
            event_count_today=len(calendar_events)
            if isinstance(calendar_events, list)
            else None,
            total_participant_count=CalendarIntelligenceProcessor._sum_participants(calendar_events),
            events_with_location_count=CalendarIntelligenceProcessor._count_with_location(calendar_events),
            unique_locations=CalendarIntelligenceProcessor._unique_locations(calendar_events),
        )

    @staticmethod
    def _safe_get_float(metrics: Optional[Dict[str, Any]], key: str) -> Optional[float]:
        if not isinstance(metrics, dict):
            return None
        v = metrics.get(key)
        if v is None:
            v = metrics.get(f"calendar_metrics_{key}")
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_get_bool(metrics: Optional[Dict[str, Any]], key: str) -> Optional[bool]:
        if not isinstance(metrics, dict):
            return None
        v = metrics.get(key)
        if v is None:
            v = metrics.get(f"calendar_metrics_{key}")
        if v is None:
            return None
        return bool(v)

    @staticmethod
    def _sum_participants(events: Optional[List[Dict[str, Any]]]) -> Optional[int]:
        if not isinstance(events, list) or not events:
            return None
        total = 0
        saw_any = False
        for e in events:
            if not isinstance(e, dict):
                continue
            p = e.get("participants")
            if isinstance(p, list) and p:
                total += len(p)
                saw_any = True
        return total if saw_any else 0 if total == 0 else total

    @staticmethod
    def _count_with_location(events: Optional[List[Dict[str, Any]]]) -> Optional[int]:
        if not isinstance(events, list) or not events:
            return None
        n = sum(
            1
            for e in events
            if isinstance(e, dict) and str(e.get("location") or "").strip()
        )
        return n

    @staticmethod
    def _unique_locations(events: Optional[List[Dict[str, Any]]]) -> List[str]:
        if not isinstance(events, list) or not events:
            return []
        seen: List[str] = []
        for e in events:
            if not isinstance(e, dict):
                continue
            loc = str(e.get("location") or "").strip()
            if loc and loc not in seen:
                seen.append(loc)
        return seen

    @staticmethod
    def _parse_hhmm(value: Any) -> Optional[tuple[int, int]]:
        if not value or not isinstance(value, str):
            return None
        try:
            parts = value.strip().split(":")
            return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
        except Exception:
            return None

    @staticmethod
    def _resolve_work_free_horizon(
        now: datetime, time_data: Dict[str, Any]
    ) -> Optional[datetime]:
        
        td = time_data or {}
        active_start = CalendarIntelligenceProcessor._parse_hhmm(
            td.get("active_hours_start_time")
            or td.get(TimeDataConstants.KEY_ACTIVE_START_STR)
        )
        active_end = CalendarIntelligenceProcessor._parse_hhmm(
            td.get("active_hours_end_time")
            or td.get(TimeDataConstants.KEY_ACTIVE_END_STR)
        )
        bedtime = CalendarIntelligenceProcessor._parse_hhmm(
            td.get(TimeDataConstants.KEY_BEDTIME_START_STR)
            or td.get("bedtime_start")
        )

        if active_start is None:
            active_start = (
                CalendarConfig.DEFAULT_ACTIVE_START_HOUR,
                CalendarConfig.DEFAULT_ACTIVE_START_MINUTE,
            )
        if active_end is None:
            active_end = (
                CalendarConfig.DEFAULT_ACTIVE_END_HOUR,
                CalendarConfig.DEFAULT_ACTIVE_END_MINUTE,
            )

        ash, asm = active_start
        eh, em = active_end
        horizon = now.replace(hour=eh, minute=em, second=0, microsecond=0)
        if horizon <= now:
            horizon = horizon + timedelta(days=1)

        if bedtime is None:
            bh, bm = (23, 0)
            bedtime_dt = now.replace(hour=bh, minute=bm, second=0, microsecond=0)
            if bedtime_dt <= now:
                bedtime_dt = bedtime_dt + timedelta(days=1)
        else:
            bh, bm = bedtime
            bedtime_dt = now.replace(hour=bh, minute=bm, second=0, microsecond=0)
            if bedtime_dt <= now:
                bedtime_dt = bedtime_dt + timedelta(days=1)
        if now < bedtime_dt <= horizon:
            horizon = bedtime_dt

        if horizon <= now:
            # Past active hours / bedtime — do NOT invent a free window until midnight
            return None

        return horizon

    @staticmethod
    def _compute_free_windows(
        events: List[Dict[str, Any]], now: datetime, time_data: Dict[str, Any]
    ) -> List[FreeWindow]:
        # ── Parse all events (future + past), normalize to user tz ──────────
        all_events: List[tuple] = []  # (start, end)
        for e in events:
            try:
                s_str = _event_start_str(e)
                e_str = _event_end_str(e)
                if not s_str or not e_str:
                    continue

                start = _ensure_comparable(dateutil.parser.isoparse(s_str), now)
                end = _ensure_comparable(dateutil.parser.isoparse(e_str), now)
                all_events.append((start, end))
            except Exception as ex:
                logger.error(f"Failed to parse event payload: {e}. Error: {ex}")
                continue

        # ── Resolve wake_time_today (anchor for empty-day case) ────────────
        td = time_data or {}
        wake = CalendarIntelligenceProcessor._parse_hhmm(
            td.get("wake_time")
            or td.get(HealthDataConstants.KEY_LAST_WAKE_TIME)
            or td.get(HealthDataConstants.KEY_WAKE_TIME)
        )
        if wake is None:
            wake = (8, 0)  # default wake at 08:00 local
        wake_h, wake_m = wake
        wake_today = now.replace(
            hour=wake_h, minute=wake_m, second=0, microsecond=0
        )
        if wake_today > now:
            # Not yet at wake — do not invent a midnight→bedtime window.
            return []

        # ── Resolve bedtime horizon (gate) ──────────────────────────────────
        bedtime = CalendarIntelligenceProcessor._parse_hhmm(
            td.get(TimeDataConstants.KEY_BEDTIME_START_STR)
            or td.get("bedtime_start")
        )
        if bedtime is None:
            bh, bm = (23, 0)
        else:
            bh, bm = bedtime
        bedtime_today = now.replace(hour=bh, minute=bm, second=0, microsecond=0)
        if bedtime_today <= now:
            bedtime_today = bedtime_today + timedelta(days=1)

        # After bedtime + 60 min buffer → no window
        if now > bedtime_today + timedelta(hours=1):
            return []

        # ── Compute CURRENT free window ─────────────────────────────────────
        future_events = sorted(
            [(s, e) for s, e in all_events if e > now], key=lambda x: x[0]
        )
        past_events_today = [
            (s, e)
            for s, e in all_events
            if e <= now and s.date() == now.date()
        ]

        free_windows: List[FreeWindow] = []

        def _make_window(
            start: datetime, end: datetime, capped_by: str, position: str
        ) -> Optional[FreeWindow]:
            gap_mins = int((end - start).total_seconds() / 60)
            if gap_mins < CalendarConfig.MIN_GAP_MINUTES:
                return None
            return FreeWindow(
                start_time=start.isoformat(),
                duration_mins=gap_mins,
                position=position,
                ends_at=end.isoformat(),
                capped_by=capped_by,
            )

        position_now = _label_for_hour(now.hour)

        if future_events:
            # Current free window: now → next event
            next_start = future_events[0][0]
            w = _make_window(
                start=now,
                end=next_start,
                capped_by="next_event",
                position=position_now,
            )
            if w is not None:
                free_windows.append(w)
            return free_windows

        if past_events_today:
            capped_by = "since_last_event"
        else:
            capped_by = "since_wake"
        start = now

        end = bedtime_today

        w = _make_window(
            start=start,
            end=end,
            capped_by=capped_by,
            position=position_now,
        )
        if w is not None:
            free_windows.append(w)
        return free_windows

    @staticmethod
    def _compute_event_states(
        events: List[Dict[str, Any]], now: datetime
    ) -> tuple[Optional[NextEvent], Optional[CurrentEvent], Optional[PreviousEvent]]:
        next_evt = None
        current_evt = None
        prev_evt = None

        min_away = float("inf")
        min_since = float("inf")

        for e in events:
            try:
                start_str = _event_start_str(e)
                end_str = _event_end_str(e)
                if not start_str:
                    continue
                start = dateutil.parser.isoparse(start_str)

                # If end_str is empty/missing, default to 30 mins after start
                if end_str:
                    end = dateutil.parser.isoparse(end_str)
                else:
                    end = start + timedelta(minutes=30)

                title = _event_title(e)
                category = _event_category(e)
                event_type = _event_type(e)

                start = _ensure_comparable(start, now)
                end = _ensure_comparable(end, now)

                if start > now:
                    mins_away = int((start - now).total_seconds() / 60)
                    if mins_away < min_away:
                        min_away = mins_away
                        next_evt = NextEvent(
                            title=title,
                            start_time=start_str,
                            minutes_away=mins_away,
                            category=category,
                            event_type=event_type,
                        )

                elif start <= now < end:
                    mins_left = int((end - now).total_seconds() / 60)
                    # Only take the first ongoing event
                    if current_evt is None:
                        current_evt = CurrentEvent(
                            title=title,
                            start_time=start_str,
                            end_time=end_str if end_str else start_str,
                            minutes_left=mins_left,
                            category=category,
                            event_type=event_type,
                        )

                elif end <= now:
                    mins_since = int((now - end).total_seconds() / 60)
                    if mins_since < min_since:
                        min_since = mins_since
                        prev_evt = PreviousEvent(
                            title=title,
                            end_time=end_str if end_str else start_str,
                            minutes_since=mins_since,
                            category=category,
                            event_type=event_type,
                        )

            except Exception as ex:
                logger.error(
                    f"Failed to parse event payload for event states: {e}. Error: {ex}"
                )
                continue

        return next_evt, current_evt, prev_evt

    @staticmethod
    def _compute_upcoming_briefs(
        events: List[Dict[str, Any]],
        now: datetime,
        limit: int = 6,
    ) -> List[CalendarEventBrief]:
        """Upcoming events with title + category for Narrative AI."""
        briefs: List[CalendarEventBrief] = []
        for e in events:
            if not isinstance(e, dict):
                continue
            start_str = _event_start_str(e)
            end_str = _event_end_str(e)
            if not start_str:
                continue
            try:
                start = _ensure_comparable(dateutil.parser.isoparse(start_str), now)
                if start <= now:
                    continue
                mins_away = int((start - now).total_seconds() / 60)
                briefs.append(
                    CalendarEventBrief(
                        title=_event_title(e),
                        start_time=start_str,
                        end_time=end_str or None,
                        minutes_away=mins_away,
                        category=_event_category(e),
                        event_type=_event_type(e),
                    )
                )
            except Exception:
                continue
        briefs.sort(key=lambda b: b.minutes_away if b.minutes_away is not None else 10**9)
        return briefs[:limit]

    @staticmethod
    def _compute_meeting_breakdown(
        events: List[Dict[str, Any]], current_date: str
    ) -> MeetingTypeBreakdown:
        focus_blocks = 0
        social = 0
        admin = 0
        other = 0
        by_category: Dict[str, int] = {}

        # Map API taxonomy → coarse buckets used by narrative
        _FOCUS_CATS = {"FOCUS", "EXECUTION", "STRATEGY", "GROWTH", "LEARNING"}
        _SOCIAL_CATS = {"SOCIAL", "PEOPLE", "LEISURE", "CELEBRATION", "BIRTHDAY", "ANNIVERSARY"}
        _ADMIN_CATS = {"ADMIN", "HOME", "TRAVEL"}
        _MEETING_CATS = {"MEETINGS", "WORK"}

        for e in events:
            # Scope filter: only include events for today
            s_str = _event_start_str(e)
            if not s_str or not s_str.startswith(current_date):
                continue

            title = _event_title(e).lower()
            category = _event_category(e) or "OTHER"
            by_category[category] = by_category.get(category, 0) + 1

            if category in _FOCUS_CATS:
                focus_blocks += 1
            elif category in _SOCIAL_CATS:
                social += 1
            elif category in _ADMIN_CATS:
                admin += 1
            elif category in _MEETING_CATS:
                if any(k in title for k in CalendarConfig.MEETING_KEYWORDS_FOCUS):
                    focus_blocks += 1
                elif any(k in title for k in CalendarConfig.MEETING_KEYWORDS_ADMIN):
                    admin += 1
                elif any(k in title for k in CalendarConfig.MEETING_KEYWORDS_SOCIAL):
                    social += 1
                else:
                    other += 1
            elif category in {"HEALTH", "FITNESS", "MEAL", "BEDTIME", "LIFESTYLE", "TIME_OFF", "HOLIDAY"}:
                other += 1
            else:
                # Fallback to pure keyword if category is OTHER / unknown
                if not title:
                    other += 1
                elif any(k in title for k in CalendarConfig.MEETING_KEYWORDS_FOCUS):
                    focus_blocks += 1
                elif any(k in title for k in CalendarConfig.MEETING_KEYWORDS_SOCIAL):
                    social += 1
                elif any(k in title for k in CalendarConfig.MEETING_KEYWORDS_ADMIN):
                    admin += 1
                else:
                    other += 1

        return MeetingTypeBreakdown(
            focus_blocks=focus_blocks,
            social=social,
            admin=admin,
            other=other,
            by_category=by_category,
        )

    @staticmethod
    def _compute_recent_meeting_count(
        events: List[Dict[str, Any]], now: datetime
    ) -> int:
        from datetime import timedelta

        two_hours_ago = now - timedelta(hours=2)
        count = 0

        for e in events:
            if e.get("allDay", False):
                continue
            s_str = _event_start_str(e)
            if not s_str:
                continue
            try:
                start = _ensure_comparable(dateutil.parser.isoparse(s_str), now)
                e_str = _event_end_str(e)
                end = (
                    dateutil.parser.isoparse(e_str) if e_str else start + timedelta(minutes=30)
                )
                end = _ensure_comparable(end, now)

                # Event overlaps with the past 2 hours
                if start <= now and end >= two_hours_ago:
                    count += 1
            except Exception:
                pass

        return count

    @staticmethod
    def _compute_meeting_density(
        events: List[Dict[str, Any]],
        current_date: str,
        now: Optional[datetime] = None,
    ) -> tuple[int, int]:
        # Work density metrics — count work/meeting-like categories
        _WORK_CATS = {"WORK", "MEETING", "MEETINGS", "FOCUS", "EXECUTION", "STRATEGY", "ADMIN"}

        today_events = []
        for e in events:
            if e.get("allDay", False):
                continue
            s_str = _event_start_str(e)
            if not s_str or not s_str.startswith(current_date):
                continue
            try:
                start = dateutil.parser.isoparse(s_str)
                e_str = _event_end_str(e)
                end = (
                    dateutil.parser.isoparse(e_str) if e_str else start + timedelta(minutes=30)
                )
                if now is not None:
                    start = _ensure_comparable(start, now)
                    end = _ensure_comparable(end, now)
                    if start > now:
                        continue
                    if end > now:
                        end = now
                category = (_event_category(e) or "").upper()
                today_events.append((start, end, category))
            except Exception:
                pass

        today_events.sort(key=lambda x: x[0])

        # Filter to work-only events for density computation
        work_events = [(s, e) for s, e, cat in today_events if cat in _WORK_CATS]
        if not work_events:
            return 0, 0

        b2b_count = 0
        max_continuous = 0
        current_b2b = 0
        current_continuous = 0
        last_end = None

        for start, end in work_events:
            duration = int((end - start).total_seconds() / 60)
            if duration <= 0:
                continue

            if last_end is None:
                current_b2b = 0
                current_continuous = duration
            else:
                gap = int((start - last_end).total_seconds() / 60)
                if gap <= 5:
                    current_b2b += 1
                    current_continuous += duration + max(0, gap)
                else:
                    b2b_count = max(b2b_count, current_b2b)
                    max_continuous = max(max_continuous, current_continuous)
                    current_b2b = 0
                    current_continuous = duration

            last_end = max(last_end, end) if last_end else end

        b2b_count = max(b2b_count, current_b2b)
        max_continuous = max(max_continuous, current_continuous)

        return b2b_count, max_continuous

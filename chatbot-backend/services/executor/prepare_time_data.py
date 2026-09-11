"""Prepare Time Data for Insight Analysis"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from services.executor.constant import APIResponseKeys, TargetKeys, TimeDataConstants
from utils.logger import logger


class PrepareTimeData:
    """Class để tính toán và trả về time data (gộp time_context và time_metrics)"""

    def __init__(
        self,
        user_profile: Optional[Dict[str, Any]] = None,
        current_dt: Optional[datetime] = None,
        tz: Optional[ZoneInfo] = None,
        timezone: Optional[str] = None,
        calendar_events: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Initialize PrepareTimeData.

        Args:
            user_profile: User profile data
            timezone: Optional timezone
            calendar_events: Optional calendar events for time metrics calculation
        """
        self.user_profile = user_profile
        self.current_dt = current_dt
        self.tz = tz
        if timezone:
            self.timezone = timezone
        elif isinstance(tz, ZoneInfo):
            self.timezone = str(tz)
        elif tz is not None:
            try:
                self.timezone = str(ZoneInfo(tz))
            except Exception:
                self.timezone = None
        else:
            from utils.logger import logger as _logger
            _logger.warning(
                "PrepareTimeData: no timezone/tz provided. "
                "Downstream 'now' will be naive unless caller attaches tz to current_dt."
            )
            self.timezone = None
        self.calendar_events = calendar_events or []
        self.time_data = None

        if self.current_dt is not None and self.current_dt.tzinfo is None:
            tz_obj = None
            if isinstance(self.tz, ZoneInfo):
                tz_obj = self.tz
            elif self.timezone:
                try:
                    tz_obj = ZoneInfo(self.timezone)
                except Exception:
                    tz_obj = None
            if tz_obj is not None:
                self.current_dt = self.current_dt.replace(tzinfo=tz_obj)
            else:
                from utils.logger import logger as _logger
                _logger.warning(
                    "PrepareTimeData: current_dt is naive and no tz supplied — "
                    "emitting naive current_time_iso. Caller MUST pass timezone."
                )

    def prepare_all_time_data(self) -> Dict[str, Any]:
        """
        Tính toán và trả về tất cả time data (gộp time_context và time_metrics).

        Returns:
            Dict duy nhất chứa tất cả thông tin:
            - time_context fields: current_dt, current_hour, current_minute, is_weekend, etc.
            - time_metrics fields: time_to_bedtime, time_to_next_event, in_active_window, free_slot_length
        """
        # Parse time context
        time_context = self._parse_time_context()

        # Calculate time metrics (requires time_context)
        time_metrics: Dict[str, Any] = {}
        if time_context:
            current_dt = time_context.get(TimeDataConstants.KEY_CURRENT_DT)
            if not current_dt:
                current_dt = self._safe_now()
            time_metrics = self._calculate_time_metrics(current_dt, time_context)
        else:
            current_dt = self._safe_now()
            time_metrics = self._calculate_time_metrics(current_dt, None)

        # Merge time_context and time_metrics into a single dict
        if time_context:
            self.time_data = {**time_context, **time_metrics}
        else:
            self.time_data = time_metrics or {}

        return self.time_data

    def _parse_time_context(self) -> Optional[Dict[str, Any]]:
        """Parse current time and extract user profile targets.

        Returns:
            Dict with current_dt, current_hour, current_minute, is_weekend,
            bedtime_start_str, bedtime_end_str, active_start_str, active_end_str, sleep_target,
            timezone, current_date, current_time_iso
        """
        try:

            current_hour = self.current_dt.hour
            current_minute = self.current_dt.minute
            is_weekend = self.current_dt.weekday() >= 5  # Saturday = 5, Sunday = 6
            current_date = self.current_dt.date()
            current_time_iso = self.current_dt.isoformat()

            # Extract user profile targets
            logger.info(f"user_profile: {self.user_profile}")
            targets = (
                self.user_profile.get(TargetKeys.TARGETS, {})
                if self.user_profile
                else {}
            )
            sleep_target = targets.get(TargetKeys.TARGET_SLEEP_HOURS, {})
            bedtime_start_str = (
                self.user_profile.get(TargetKeys.BEDTIME_START)
                if self.user_profile
                else None
            )
            bedtime_end_str = (
                self.user_profile.get(TargetKeys.BEDTIME_END)
                if self.user_profile
                else None
            )
            active_start_str = (
                self.user_profile.get(TargetKeys.ACTIVE_HOURS_START_TIME)
                if self.user_profile
                else None
            )
            active_end_str = (
                self.user_profile.get(TargetKeys.ACTIVE_HOURS_END_TIME)
                if self.user_profile
                else None
            )

            return {
                TimeDataConstants.KEY_CURRENT_DT: self.current_dt,
                TimeDataConstants.KEY_CURRENT_HOUR: current_hour,
                TimeDataConstants.KEY_CURRENT_MINUTE: current_minute,
                TimeDataConstants.KEY_IS_WEEKEND: is_weekend,
                TimeDataConstants.KEY_TZ: self.tz,
                TimeDataConstants.KEY_TIMEZONE: self.timezone,
                TimeDataConstants.KEY_CURRENT_DATE: current_date,
                TimeDataConstants.KEY_CURRENT_TIME_ISO: current_time_iso,
                TimeDataConstants.KEY_BEDTIME_START_STR: bedtime_start_str,
                TimeDataConstants.KEY_BEDTIME_END_STR: bedtime_end_str,
                TimeDataConstants.KEY_ACTIVE_START_STR: active_start_str,
                TimeDataConstants.KEY_ACTIVE_END_STR: active_end_str,
                TimeDataConstants.KEY_SLEEP_TARGET: sleep_target,
            }
        except Exception as e:
            logger.warning(f"Error parsing time context: {str(e)}")
            return None

    def _calculate_time_metrics(
        self, current_dt: datetime, time_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate time-related metrics.

        Args:
            current_dt: Current datetime
            time_context: Time context dict (can be None)

        Returns:
            Dict with time_to_bedtime, time_to_next_event, in_active_window, free_slot_length
        """
        if not time_context:
            # Return empty metrics if no time_context
            return {
                TimeDataConstants.KEY_TIME_TO_BEDTIME: None,
                TimeDataConstants.KEY_TIME_TO_NEXT_EVENT: None,
                TimeDataConstants.KEY_IN_ACTIVE_WINDOW: False,
                TimeDataConstants.KEY_FREE_SLOT_LENGTH: None,
            }

        bedtime_start_str = time_context.get(TimeDataConstants.KEY_BEDTIME_START_STR)
        active_start_str = time_context.get(TimeDataConstants.KEY_ACTIVE_START_STR)
        active_end_str = time_context.get(TimeDataConstants.KEY_ACTIVE_END_STR)
        current_hour = time_context[TimeDataConstants.KEY_CURRENT_HOUR]
        current_minute = time_context[TimeDataConstants.KEY_CURRENT_MINUTE]
        tz = time_context.get(
            TimeDataConstants.KEY_TZ,
            (
                self.tz
                if isinstance(self.tz, ZoneInfo)
                else (
                    ZoneInfo(self.timezone)
                    if self.timezone
                    else None
                )
            ),
        )

        # Calculate time_to_bedtime (minutes until bedtime_start)
        time_to_bedtime = None
        if bedtime_start_str:
            try:
                bedtime_parts = bedtime_start_str.split(":")
                bedtime_hour = int(bedtime_parts[0])
                bedtime_minute = int(bedtime_parts[1]) if len(bedtime_parts) > 1 else 0
                bedtime_today = current_dt.replace(
                    hour=bedtime_hour, minute=bedtime_minute, second=0, microsecond=0
                )

                # If bedtime has passed today, use tomorrow
                if bedtime_today < current_dt:
                    bedtime_today = bedtime_today + timedelta(days=1)

                time_to_bedtime = int((bedtime_today - current_dt).total_seconds() / 60)
            except Exception:
                pass

        # Calculate time_to_next_event (minutes until the nearest upcoming event)
        time_to_next_event = None
        if self.calendar_events:
            try:
                for event in self.calendar_events:
                    start_time_str = event.get(APIResponseKeys.START_TIME)
                    if not start_time_str:
                        continue
                    try:
                        event_start = datetime.fromisoformat(
                            start_time_str.replace("Z", "+00:00")
                        )
                        if event_start.tzinfo is None:
                            event_start = event_start.replace(tzinfo=tz)
                        else:
                            event_start = event_start.astimezone(tz)

                        if current_dt.tzinfo is None:
                            current_cmp = current_dt.replace(tzinfo=tz)
                        else:
                            current_cmp = current_dt.astimezone(tz)

                        if event_start > current_cmp:
                            mins = int(
                                (event_start - current_cmp).total_seconds() / 60
                            )
                            if time_to_next_event is None or mins < time_to_next_event:
                                time_to_next_event = mins
                    except Exception:
                        continue
            except Exception:
                pass

        # Calculate in_active_window
        in_active_window = False
        if active_start_str and active_end_str:
            try:
                active_start_parts = active_start_str.split(":")
                active_end_parts = active_end_str.split(":")
                active_start_hour = int(active_start_parts[0])
                active_start_min = (
                    int(active_start_parts[1]) if len(active_start_parts) > 1 else 0
                )
                active_end_hour = int(active_end_parts[0])
                active_end_min = (
                    int(active_end_parts[1]) if len(active_end_parts) > 1 else 0
                )

                current_time_minutes = current_hour * 60 + current_minute
                active_start_minutes = active_start_hour * 60 + active_start_min
                active_end_minutes = active_end_hour * 60 + active_end_min

                # Handle wrap-around (e.g., 22:00 to 08:00)
                if active_start_minutes > active_end_minutes:
                    in_active_window = (
                        current_time_minutes >= active_start_minutes
                    ) or (current_time_minutes <= active_end_minutes)
                else:
                    in_active_window = (
                        active_start_minutes
                        <= current_time_minutes
                        <= active_end_minutes
                    )
            except Exception:
                pass

        free_slot_length = None
        if time_to_next_event is not None:
            free_slot_length = time_to_next_event
        elif active_end_str:
            try:
                active_end_parts = active_end_str.split(":")
                active_end_hour = int(active_end_parts[0])
                active_end_min = (
                    int(active_end_parts[1]) if len(active_end_parts) > 1 else 0
                )
                active_end_today = current_dt.replace(
                    hour=active_end_hour, minute=active_end_min, second=0, microsecond=0
                )
                if active_end_today <= current_dt:
                    active_end_today = active_end_today + timedelta(days=1)
                free_slot_length = int(
                    (active_end_today - current_dt).total_seconds() / 60
                )
            except Exception:
                pass

        if (
            free_slot_length is not None
            and time_to_bedtime is not None
            and time_to_bedtime >= 0
        ):
            free_slot_length = min(free_slot_length, time_to_bedtime)

        return {
            TimeDataConstants.KEY_TIME_TO_BEDTIME: time_to_bedtime,
            TimeDataConstants.KEY_TIME_TO_NEXT_EVENT: time_to_next_event,
            TimeDataConstants.KEY_IN_ACTIVE_WINDOW: in_active_window,
            TimeDataConstants.KEY_FREE_SLOT_LENGTH: free_slot_length,
        }

    def _safe_now(self) -> datetime:
        """Return tz-aware `now()` using best-known tz, never silently UTC."""
        if isinstance(self.tz, ZoneInfo):
            return datetime.now(self.tz)
        if self.timezone:
            try:
                return datetime.now(ZoneInfo(self.timezone))
            except Exception:
                pass
        from utils.logger import logger as _logger
        _logger.warning(
            "PrepareTimeData: no tz available, returning naive datetime.now(). "
            "Pass timezone= or tz= in constructor."
        )
        return datetime.now()

    def get_time_data(self) -> Optional[Dict[str, Any]]:
        """Get cached time data (gộp time_context và time_metrics)."""
        return self.time_data

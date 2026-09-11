"""Prepare calendar data for insight analysis"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from services.executor.constant import CalendarDataConstants, TargetKeys
from services.external_api_service import IExternalAPIService
from utils.logger import logger


class PrepareCalendarData:
    """
    Class responsible for fetching and preparing calendar data for insight analysis.

    This class handles:
    - Fetching calendar events from external API (8 days: today + 7 days)
    - Formatting calendar data for LLM analysis
    - Calculating calendar metrics (back-to-back count, meeting minutes, calendar density, etc.)
    """

    def __init__(
        self,
        external_api_service: IExternalAPIService,
        user_id: str,
        provider_name: Optional[str] = "",
        timezone: Optional[str] = None,
        current_dt: Optional[datetime] = None,
        user_profile: Optional[Dict[str, Any]] = None,
        prefetched_events: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Initialize PrepareCalendarData.

        Args:
            external_api_service: External API service instance
            user_id: User ID to fetch calendar events for
            provider_name: Optional calendar provider name (default: empty string)
            timezone: Optional IANA timezone
            current_dt: Optional current datetime (if None, will be created from timezone)
            user_profile: Optional user profile data (for active hours)
            prefetched_events: Optional list of calendar events already fetched
                (e.g. from DataCollector). When provided, _fetch_calendar_events
                skips the external API call and reuses this list directly.
        """
        self.external_api_service = external_api_service
        self.user_id = user_id
        self.provider_name = provider_name or ""
        if timezone:
            self.timezone = timezone
        else:
            from utils.logger import logger as _logger
            self.timezone = "UTC"
        self.current_dt = current_dt or datetime.now(ZoneInfo(self.timezone))
        self.user_profile = user_profile or {}
        self._prefetched_events = prefetched_events

        # Cache for fetched data
        self.raw_events = None  # Raw events from API (UTC timezone)
        self.events = None  # All events (8 days) with converted timezone
        self.events_today = None  # Today's events only
        self.formatted_calendar_data = None
        self.calendar_metrics = None

    async def prepare_all_calendar_data(self) -> Dict[str, Any]:
        """
        Prepare all calendar data: fetch events (8 days), format for LLM, and calculate metrics.

        Returns:
            Dict with:
            - events: List of today's calendar events
            - calendar_events_week: All fetched events (week start through today+7)
            - calendar_events_next7d: List of events in next 7 days
            - calendar_events_next48h: List of events in next 48 hours
            - calendar_events_next3h: List of events in next 3 hours
            - formatted_calendar_data: String formatted for LLM
            - calendar_metrics: Dict with various metrics
        """
        # Fetch calendar events (8 days: today + 7 days)
        await self._fetch_calendar_events()

        # Filter events by time windows
        events_today = self._filter_events_today(self.events or [])
        events_next7d = self._filter_events_next_n_days(self.events or [], days=7)
        events_next48h = self._filter_events_next_n_hours(self.events or [], hours=48)
        events_next3h = self._filter_events_next_n_hours(self.events or [], hours=3)

        # Store today's events
        self.events_today = events_today

        # Format calendar data for LLM (only today's events)
        self.formatted_calendar_data = self._format_calendar_data_for_llm(
            events_today, timezone=self.timezone
        )

        # Calculate calendar metrics (using all events for weekly calculations)
        self.calendar_metrics = self._calculate_calendar_metrics(
            events_today, self.events or [], self.current_dt, self.timezone
        )

        logger.info(
            f"📊 Formatted {len(events_today)} today's events, {len(self.events or [])} total events (8 days) for analysis"
        )

        return {
            CalendarDataConstants.KEY_EVENTS: events_today,
            CalendarDataConstants.KEY_CALENDAR_EVENTS_WEEK: self.events or [],
            CalendarDataConstants.KEY_CALENDAR_EVENTS_NEXT7D: events_next7d,
            CalendarDataConstants.KEY_CALENDAR_EVENTS_NEXT48H: events_next48h,
            CalendarDataConstants.KEY_CALENDAR_EVENTS_NEXT3H: events_next3h,
            CalendarDataConstants.KEY_FORMATTED_CALENDAR_DATA: self.formatted_calendar_data,
            CalendarDataConstants.KEY_CALENDAR_METRICS: self.calendar_metrics,
        }

    async def _fetch_calendar_events(self) -> None:
        """Fetch calendar events from external API (8 days: today + 7 days) and convert timezone."""
        if self.raw_events is not None:
            # Already fetched, skip
            return

        # If caller already provided events (e.g. via DataCollector), reuse them
        # and skip the external API call entirely. Shape: List[Dict] with
        # startTime/endTime/summary fields, as returned by get_calendar_events().
        if self._prefetched_events is not None:
            self.raw_events = self._prefetched_events or []
            if not self.raw_events:
                self.events = []
                return
            self.events = self._convert_events_timezone(self.raw_events)
            logger.info(
                f"♻️ Reusing {len(self.raw_events)} prefetched calendar events for user {self.user_id}"
            )
            return

        try:
            # Monday 00:00 of the current week through (today + 7 days) 23:59
            tz = ZoneInfo(self.timezone)
            today_start = self.current_dt.replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            monday_start = today_start - timedelta(days=today_start.weekday())
            end_date = today_start + timedelta(days=7)
            end_date = end_date.replace(hour=23, minute=59, second=59)

            # Format dates for API: "YYYY-MM-DDTHH:mm"
            start_date_time = monday_start.strftime("%Y-%m-%dT%H:%M")
            end_date_time = end_date.strftime("%Y-%m-%dT%H:%M")
            logger.info(
                f"start_date_time: {start_date_time}, end_date_time: {end_date_time}"
            )

            # Fetch raw events from API (UTC timezone)
            self.raw_events = await self.external_api_service.get_calendar_events(
                user_id=self.user_id,
                provider_name=self.provider_name,
                timezone=self.timezone,
                start_date_time=start_date_time,
                end_date_time=end_date_time,
                include_cancelled=False,
                include_declined=False,
            )

            if self.raw_events is None:
                logger.warning(
                    f"⚠️ Failed to retrieve calendar events for user {self.user_id}"
                )
                self.raw_events = []
                self.events = []
                return

            # Convert all events' start and end times to user timezone
            self.events = self._convert_events_timezone(self.raw_events)

            logger.info(
                f"✅ Fetched {len(self.raw_events)} calendar events and converted to {self.timezone} timezone"
            )
        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch calendar events: {str(e)}")
            self.raw_events = []
            self.events = []

    def _convert_events_timezone(
        self, raw_events: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Convert all events' start and end times from UTC to user timezone.

        Args:
            raw_events: List of raw event dictionaries with UTC times

        Returns:
            List of event dictionaries with converted times in user timezone
        """
        converted_events = []
        user_tz = ZoneInfo(self.timezone)

        for event in raw_events:
            converted_event = event.copy()

            # Convert startTime
            start_time_str = event.get(CalendarDataConstants.KEY_START_TIME)
            if start_time_str:
                start_dt = self._parse_utc_datetime(start_time_str)
                if start_dt:
                    # Convert to user timezone and format back to ISO string
                    converted_event[CalendarDataConstants.KEY_START_TIME] = (
                        start_dt.astimezone(user_tz).isoformat()
                    )

            # Convert endTime
            end_time_str = event.get(CalendarDataConstants.KEY_END_TIME)
            if end_time_str:
                end_dt = self._parse_utc_datetime(end_time_str)
                if end_dt:
                    # Convert to user timezone and format back to ISO string
                    converted_event[CalendarDataConstants.KEY_END_TIME] = (
                        end_dt.astimezone(user_tz).isoformat()
                    )

            converted_events.append(converted_event)

        return converted_events

    def _parse_utc_datetime(self, datetime_str: str) -> Optional[datetime]:
        """
        Parse datetime string from UTC.

        Args:
            datetime_str: ISO format datetime string (assumed to be UTC)

        Returns:
            Datetime object with UTC timezone, or None if parsing fails
        """
        if not datetime_str:
            return None

        try:
            # Parse as UTC (replace Z with +00:00 for fromisoformat)
            dt_str = datetime_str.replace("Z", "+00:00")
            dt = datetime.fromisoformat(dt_str)

            # Ensure it has UTC timezone info
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=ZoneInfo("UTC"))

            return dt
        except Exception as e:
            logger.warning(f"⚠️ Failed to parse datetime '{datetime_str}': {str(e)}")
            return None

    def _parse_event_datetime(self, datetime_str: str) -> Optional[datetime]:
        """
        Parse event datetime string (already in user timezone after conversion).

        Args:
            datetime_str: ISO format datetime string (in user timezone, may have timezone info)

        Returns:
            Datetime object in user timezone, or None if parsing fails
        """
        if not datetime_str:
            return None

        try:
            # Parse datetime (already converted, may have timezone info like "+07:00" or "Z")
            # Handle both cases: with timezone info and without
            if "Z" in datetime_str:
                # Still has Z (shouldn't happen after conversion, but handle it)
                dt_str = datetime_str.replace("Z", "+00:00")
                dt = datetime.fromisoformat(dt_str)
                # Convert from UTC to user timezone
                user_tz = ZoneInfo(self.timezone)
                dt = dt.astimezone(user_tz)
            else:
                # Already has timezone info (e.g., "+07:00") or no timezone
                dt = datetime.fromisoformat(datetime_str)
                # If no timezone info, assume it's already in user timezone
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=ZoneInfo(self.timezone))

            return dt
        except Exception as e:
            logger.warning(f"⚠️ Failed to parse datetime '{datetime_str}': {str(e)}")
            return None

    def _filter_events_today(
        self, events: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter events for today only."""
        if not events:
            return []

        today_start = self.current_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        today_end = today_start + timedelta(days=1)

        filtered = []
        for event in events:
            start_time_str = event.get(CalendarDataConstants.KEY_START_TIME)
            if not start_time_str:
                continue

            event_start = self._parse_event_datetime(start_time_str)
            if event_start and today_start <= event_start < today_end:
                filtered.append(event)

        return filtered

    def _filter_events_next_n_days(
        self, events: List[Dict[str, Any]], days: int
    ) -> List[Dict[str, Any]]:
        """Filter events in next N days (excluding today)."""
        if not events:
            return []

        today_end = self.current_dt.replace(
            hour=23, minute=59, second=59, microsecond=999999
        )
        end_date = today_end + timedelta(days=days)

        filtered = []
        for event in events:
            start_time_str = event.get(CalendarDataConstants.KEY_START_TIME)
            if not start_time_str:
                continue

            event_start = self._parse_event_datetime(start_time_str)
            if event_start and today_end < event_start <= end_date:
                filtered.append(event)

        return filtered

    def _filter_events_next_n_hours(
        self, events: List[Dict[str, Any]], hours: int
    ) -> List[Dict[str, Any]]:
        """Filter events in next N hours."""
        if not events:
            return []

        end_time = self.current_dt + timedelta(hours=hours)

        filtered = []
        for event in events:
            start_time_str = event.get(CalendarDataConstants.KEY_START_TIME)
            if not start_time_str:
                continue

            event_start = self._parse_event_datetime(start_time_str)
            if event_start and self.current_dt < event_start <= end_time:
                filtered.append(event)

        return filtered

    def _event_category_label(self, event: Dict[str, Any]) -> Optional[str]:
        """Return normalized category from API event when present."""
        raw = event.get(CalendarDataConstants.KEY_CATEGORY)
        if raw is None or str(raw).strip() == "":
            return None
        return str(raw).strip().upper()

    def _format_calendar_data_for_llm(
        self, events: List[Dict[str, Any]], timezone: Optional[str] = None
    ) -> str:
        """Format calendar events into readable text for LLM analysis.

        Args:
            events: List of calendar event dictionaries
            timezone: Optional timezone to convert times to

        Returns:
            Formatted string representation of events
        """
        if not events:
            return CalendarDataConstants.DEFAULT_NO_EVENTS_MESSAGE

        formatted = []
        formatted.append(f"Today's Calendar Events ({len(events)} total):\n")

        for idx, event in enumerate(events, 1):
            summary = event.get(CalendarDataConstants.KEY_SUMMARY, "")
            start_time = event.get(CalendarDataConstants.KEY_START_TIME, "")
            end_time = event.get(CalendarDataConstants.KEY_END_TIME, "")
            location = event.get(CalendarDataConstants.KEY_LOCATION, "")
            description = event.get(CalendarDataConstants.KEY_DESCRIPTION, "")
            all_day = event.get(CalendarDataConstants.KEY_ALL_DAY, False)
            participants = event.get(CalendarDataConstants.KEY_PARTICIPANTS, [])
            # Convert times to readable format (already in user timezone)
            if start_time and end_time:
                start_dt = self._parse_event_datetime(start_time)
                end_dt = self._parse_event_datetime(end_time)

                if start_dt and end_dt:
                    # Format as readable time string
                    start_time = start_dt.strftime("%Y-%m-%d %H:%M:%S")
                    end_time = end_dt.strftime("%Y-%m-%d %H:%M:%S")

            event_str = f"\n{idx}. {summary}\n"
            if all_day:
                event_str += "   Type: All-day event\n"
            else:
                event_str += f"   Time: {start_time} - {end_time}\n"
            category = self._event_category_label(event)
            if category:
                event_str += f"   Category: {category}\n"
            if location:
                event_str += f"   Location: {location}\n"
            if participants:
                # Format participants: name (email)
                formatted_participants = []
                max_participants = CalendarDataConstants.MAX_PARTICIPANTS_TO_SHOW
                for participant in participants[:max_participants]:
                    if isinstance(participant, dict):
                        name = participant.get(CalendarDataConstants.KEY_NAME, "")
                        email = participant.get(CalendarDataConstants.KEY_EMAIL, "")
                        if name and email:
                            formatted_participants.append(f"{name} ({email})")
                        elif email:
                            formatted_participants.append(email)
                        elif name:
                            formatted_participants.append(name)
                    elif isinstance(participant, str):
                        formatted_participants.append(participant)

                participants_str = ", ".join(formatted_participants)

                # If more than max_participants, add "and X more"
                if len(participants) > max_participants:
                    remaining_count = len(participants) - max_participants
                    if remaining_count > 0:
                        participants_str += f", and {remaining_count} more"

                event_str += f"   Participants: {participants_str}\n"
            if description:
                event_str += f"   Description: {description}\n"

            formatted.append(event_str)

        return "".join(formatted)

    def _calculate_calendar_metrics(
        self,
        events_today: List[Dict[str, Any]],
        events_all: List[Dict[str, Any]],
        current_dt: datetime,
        timezone: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate calendar-related metrics.

        Args:
            events_today: Today's calendar events
            events_all: All calendar events (8 days)
            current_dt: Current datetime
            timezone: Optional timezone

        Returns:
            Dict with various calendar metrics
        """
        back_to_back_count = 0
        meeting_minutes = 0
        calendar_density = 0
        continuous_events_minutes = 0
        work_events_hours_weekend = 0.0
        work_load_high = False
        weekly_work_hours = 0.0

        if not events_today and not events_all:
            return {
                CalendarDataConstants.KEY_BACK_TO_BACK_COUNT: 0,
                CalendarDataConstants.KEY_MEETING_MINUTES: 0,
                CalendarDataConstants.KEY_CALENDAR_DENSITY: 0,
                CalendarDataConstants.KEY_CONTINUOUS_EVENTS_MINUTES: 0,
                CalendarDataConstants.KEY_WORK_EVENTS_HOURS_WEEKEND: 0.0,
                CalendarDataConstants.KEY_WORK_LOAD_HIGH: False,
                CalendarDataConstants.KEY_WORK_EVENTS_HOURS_WEEK: 0.0,
            }

        # Note: timezone conversion is handled by _parse_event_datetime

        # Get active hours from user profile
        active_hours_start_str = self.user_profile.get(
            TargetKeys.ACTIVE_HOURS_START_TIME
        )
        active_hours_end_str = self.user_profile.get(TargetKeys.ACTIVE_HOURS_END_TIME)

        # Calculate back-to-back count
        for i in range(len(events_today) - 1):
            event1 = events_today[i]
            event2 = events_today[i + 1]
            end_time1_str = event1.get(CalendarDataConstants.KEY_END_TIME)
            start_time2_str = event2.get(CalendarDataConstants.KEY_START_TIME)

            if end_time1_str and start_time2_str:
                end1 = self._parse_event_datetime(end_time1_str)
                start2 = self._parse_event_datetime(start_time2_str)

                if end1 and start2:
                    # Check if events are back-to-back (no gap or very small gap < threshold)
                    gap_minutes = (start2 - end1).total_seconds() / 60
                    threshold = CalendarDataConstants.BACK_TO_BACK_GAP_THRESHOLD_MINUTES
                    # Overlaps (negative gap) count as back-to-back, same as a
                    # zero/sub-threshold gap. The previous `0 <= gap` bound
                    # dropped overlapping pairs like 08:30–09:30 vs 09:00–10:00.
                    if gap_minutes < threshold:
                        back_to_back_count += 1

        # Meeting minutes across today's full day.
        try:
            day_start = current_dt.replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            day_end = day_start + timedelta(days=1)
            for event in events_today:
                start_time_str = event.get(CalendarDataConstants.KEY_START_TIME)
                end_time_str = event.get(CalendarDataConstants.KEY_END_TIME)
                if start_time_str and end_time_str:
                    event_start = self._parse_event_datetime(start_time_str)
                    event_end = self._parse_event_datetime(end_time_str)

                    if event_start and event_end:
                        # Clip event to today's bounds
                        overlap_start = max(event_start, day_start)
                        overlap_end = min(event_end, day_end)
                        if overlap_end > overlap_start:
                            meeting_minutes += (
                                overlap_end - overlap_start
                            ).total_seconds() / 60
        except Exception:
            pass

        # Calculate calendar_density (number of meetings in active hours)
        if active_hours_start_str and active_hours_end_str:
            try:
                # Parse active hours
                active_start_parts = active_hours_start_str.split(":")
                active_end_parts = active_hours_end_str.split(":")
                active_start_hour = int(active_start_parts[0])
                active_start_min = (
                    int(active_start_parts[1]) if len(active_start_parts) > 1 else 0
                )
                active_end_hour = int(active_end_parts[0])
                active_end_min = (
                    int(active_end_parts[1]) if len(active_end_parts) > 1 else 0
                )

                active_start_minutes = active_start_hour * 60 + active_start_min
                active_end_minutes = active_end_hour * 60 + active_end_min

                for event in events_today:
                    start_time_str = event.get(CalendarDataConstants.KEY_START_TIME)
                    if not start_time_str:
                        continue

                    event_start = self._parse_event_datetime(start_time_str)
                    if event_start:
                        event_hour = event_start.hour
                        event_min = event_start.minute
                        event_time_minutes = event_hour * 60 + event_min

                        # Check if event is within active hours (handle wrap-around)
                        in_active_hours = False
                        if active_start_minutes < active_end_minutes:
                            # Normal case: e.g., 09:00 to 17:00
                            in_active_hours = (
                                active_start_minutes
                                <= event_time_minutes
                                < active_end_minutes
                            )
                        else:
                            # Wrap-around case: e.g., 22:00 to 08:00
                            in_active_hours = (
                                event_time_minutes >= active_start_minutes
                            ) or (event_time_minutes < active_end_minutes)

                        if in_active_hours:
                            calendar_density += 1
            except Exception as e:
                logger.warning(f"⚠️ Failed to calculate calendar_density: {str(e)}")

        # Calculate continuous_events_minutes (continuous work >=90 mins)
        try:
            threshold_minutes = (
                CalendarDataConstants.CONTINUOUS_EVENTS_THRESHOLD_MINUTES
            )
            sorted_events = sorted(
                events_today,
                key=lambda x: x.get(CalendarDataConstants.KEY_START_TIME, ""),
            )

            if len(sorted_events) > 0:
                current_chain_start = None
                current_chain_end = None
                max_continuous_minutes = 0

                for event in sorted_events:
                    start_time_str = event.get(CalendarDataConstants.KEY_START_TIME)
                    end_time_str = event.get(CalendarDataConstants.KEY_END_TIME)
                    if not start_time_str or not end_time_str:
                        continue

                    event_start = self._parse_event_datetime(start_time_str)
                    event_end = self._parse_event_datetime(end_time_str)

                    if not event_start or not event_end:
                        continue

                    if current_chain_start is None:
                        # Start new chain
                        current_chain_start = event_start
                        current_chain_end = event_end
                    else:
                        # Check if event continues the chain (gap < 5 mins)
                        gap_minutes = (
                            event_start - current_chain_end
                        ).total_seconds() / 60
                        if (
                            0
                            <= gap_minutes
                            < CalendarDataConstants.BACK_TO_BACK_GAP_THRESHOLD_MINUTES
                        ):
                            # Extend chain
                            current_chain_end = event_end
                        else:
                            # End current chain, check if it meets threshold
                            chain_duration = (
                                current_chain_end - current_chain_start
                            ).total_seconds() / 60
                            if chain_duration >= threshold_minutes:
                                max_continuous_minutes = max(
                                    max_continuous_minutes, chain_duration
                                )
                            # Start new chain
                            current_chain_start = event_start
                            current_chain_end = event_end

                # Check last chain
                if current_chain_start and current_chain_end:
                    chain_duration = (
                        current_chain_end - current_chain_start
                    ).total_seconds() / 60
                    if chain_duration >= threshold_minutes:
                        max_continuous_minutes = max(
                            max_continuous_minutes, chain_duration
                        )

                continuous_events_minutes = int(max_continuous_minutes)
        except Exception as e:
            logger.warning(f"⚠️ Failed to calculate continuous_events_minutes: {str(e)}")

        # Calculate work_events_hours_weekend (Sat–Sun of the current Mon-based week)
        try:
            current_weekday = current_dt.weekday()  # Monday=0, Sunday=6
            monday_start = current_dt.replace(
                hour=0, minute=0, second=0, microsecond=0
            ) - timedelta(days=current_weekday)
            saturday_start = monday_start + timedelta(days=5)
            sunday_end = monday_start + timedelta(days=6, hours=23, minutes=59)

            for event in events_all:
                start_time_str = event.get(CalendarDataConstants.KEY_START_TIME)
                end_time_str = event.get(CalendarDataConstants.KEY_END_TIME)
                if not start_time_str or not end_time_str:
                    continue

                event_start = self._parse_event_datetime(start_time_str)
                event_end = self._parse_event_datetime(end_time_str)

                if event_start and event_end:
                    # Check if event overlaps with weekend
                    if event_end > saturday_start and event_start < sunday_end:
                        overlap_start = max(event_start, saturday_start)
                        overlap_end = min(event_end, sunday_end)
                        work_events_hours_weekend += (
                            overlap_end - overlap_start
                        ).total_seconds() / 3600  # Convert to hours
        except Exception as e:
            logger.warning(f"⚠️ Failed to calculate work_events_hours_weekend: {str(e)}")

        # Calculate work_load_high (weekly work hours > threshold)
        try:
            # Get Monday of current week
            current_weekday = current_dt.weekday()  # Monday=0
            days_since_monday = current_weekday
            monday_start = current_dt.replace(
                hour=0, minute=0, second=0, microsecond=0
            ) - timedelta(days=days_since_monday)

            weekly_work_hours = 0.0
            # Monday 00:00 through next Monday 00:00 (exclusive). Remaining
            # current-week events (e.g. tonight's 19:30 meeting) are included;
            # cutting at current_dt under-counted work_events_hours_week.
            week_end = monday_start + timedelta(days=7)
            for event in events_all:
                start_time_str = event.get(CalendarDataConstants.KEY_START_TIME)
                end_time_str = event.get(CalendarDataConstants.KEY_END_TIME)
                if not start_time_str or not end_time_str:
                    continue

                event_start = self._parse_event_datetime(start_time_str)
                event_end = self._parse_event_datetime(end_time_str)

                if event_start and event_end:
                    if monday_start <= event_start < week_end:
                        event_duration_hours = (
                            event_end - event_start
                        ).total_seconds() / 3600
                        weekly_work_hours += event_duration_hours

            threshold_hours = CalendarDataConstants.WORK_LOAD_HIGH_THRESHOLD_HOURS
            work_load_high = weekly_work_hours > threshold_hours
        except Exception as e:
            logger.warning(f"⚠️ Failed to calculate work_load_high: {str(e)}")

        return {
            CalendarDataConstants.KEY_BACK_TO_BACK_COUNT: back_to_back_count,
            CalendarDataConstants.KEY_MEETING_MINUTES: meeting_minutes,
            CalendarDataConstants.KEY_CALENDAR_DENSITY: calendar_density,
            CalendarDataConstants.KEY_CONTINUOUS_EVENTS_MINUTES: continuous_events_minutes,
            CalendarDataConstants.KEY_WORK_EVENTS_HOURS_WEEKEND: round(
                work_events_hours_weekend, 2
            ),
            CalendarDataConstants.KEY_WORK_LOAD_HIGH: work_load_high,
            CalendarDataConstants.KEY_WORK_EVENTS_HOURS_WEEK: round(
                weekly_work_hours, 2
            ),
        }

    def get_events(self) -> List[Dict[str, Any]]:
        """Get cached calendar events (today only)."""
        return self.events_today or []

    def get_formatted_calendar_data(self) -> str:
        """Get formatted calendar data for LLM."""
        return (
            self.formatted_calendar_data
            or CalendarDataConstants.DEFAULT_NO_EVENTS_MESSAGE
        )

    def get_calendar_metrics(self) -> Dict[str, Any]:
        """Get calculated calendar metrics."""
        return self.calendar_metrics or {
            CalendarDataConstants.KEY_BACK_TO_BACK_COUNT: 0,
            CalendarDataConstants.KEY_MEETING_MINUTES: 0,
            CalendarDataConstants.KEY_CALENDAR_DENSITY: 0,
            CalendarDataConstants.KEY_CONTINUOUS_EVENTS_MINUTES: 0,
            CalendarDataConstants.KEY_WORK_EVENTS_HOURS_WEEKEND: 0.0,
            CalendarDataConstants.KEY_WORK_LOAD_HIGH: False,
            CalendarDataConstants.KEY_WORK_EVENTS_HOURS_WEEK: 0.0,
        }

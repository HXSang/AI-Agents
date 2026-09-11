import calendar
import copy
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from dateutil.parser import isoparse
from insights.insight_config import UrgencyLevel
from insights.insight_config import GoalConfig
from insights.schemas.processed_context import CalendarIntelligenceBlock
from insights.schemas.processed_context import FreeWindow
from insights.schemas.processed_context import GoalSignal
from insights.schemas.processed_context import GoalSignalsBlock
from insights.schemas.processed_context import HealthSignalsBlock



class AvailablePool:
    def __init__(self, free_windows: List[FreeWindow]):
        self.windows = copy.deepcopy(free_windows)

    def allocate(
        self, needed_mins: int, min_usable_threshold: int = 20, buffer_mins: int = 15
    ) -> Optional[FreeWindow]:
        for i, fw in enumerate(self.windows):
            if fw.duration_mins >= needed_mins:
                # Can allocate from this window
                allocated = FreeWindow(
                    start_time=fw.start_time,
                    duration_mins=needed_mins,
                    position=fw.position,
                )

                consumed = needed_mins + buffer_mins
                fw.duration_mins -= consumed

                if fw.duration_mins > 0:
                    try:
                        dt = isoparse(fw.start_time)
                        new_start = dt + timedelta(minutes=consumed)
                        fw.start_time = new_start.isoformat()
                    except Exception:
                        pass

                if fw.duration_mins < min_usable_threshold:
                    self.windows.pop(i)

                return allocated
        return None


class GoalProgressProcessor:
    @staticmethod
    def process(
        health_signals: Optional[HealthSignalsBlock],
        calendar_intelligence: CalendarIntelligenceBlock,
        user_goals: Optional[List[Dict[str, Any]]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> GoalSignalsBlock:

        goals = []
        pool = AvailablePool(calendar_intelligence.free_windows_today)

        # Priority 1: Sleep (Severe deficit usually requires a nap)
        if (
            health_signals
            and health_signals.sleep
            and health_signals.sleep.overall.status != "insufficient_data"
            and health_signals.sleep.overall.status != "no_data"
        ):
            goals.append(
                GoalProgressProcessor._process_daily_sleep(
                    sleep=health_signals.sleep, pool=pool
                )
            )

        # Priority 2: Steps
        if (
            health_signals
            and health_signals.activity
            and health_signals.activity.level != "unavailable"
        ):
            goals.append(
                GoalProgressProcessor._process_daily_steps(
                    steps=health_signals.activity, pool=pool
                )
            )

        # 2. Process Tier 2 Goals (Custom User Goals)
        if user_goals:
            for g in user_goals:
                custom_sig = GoalProgressProcessor._process_custom_goal(
                    user_goal=g,
                    pool=pool,
                    health_signals=health_signals,
                    user_profile=user_profile,
                )
                if custom_sig:
                    goals.append(custom_sig)

        return GoalSignalsBlock(goals=goals)

    @staticmethod
    def _process_daily_steps(steps, pool: AvailablePool) -> GoalSignal:

        pace_ratio = steps.pace_ratio if steps.pace_ratio is not None else 0.0
        steps_goal_val = steps.steps_goal if steps.steps_goal is not None else 0
        steps_today_val = steps.steps_today if steps.steps_today is not None else 0
        daily_needed = max(0, steps_goal_val - steps_today_val)
        days_remaining = 0

        t = GoalConfig.STEPS_PACE_THRESHOLDS
        if pace_ratio >= t[0]:
            urgency = UrgencyLevel.ON_TRACK
        elif pace_ratio >= t[1]:
            urgency = UrgencyLevel.LOW
        elif pace_ratio >= t[2]:
            urgency = UrgencyLevel.MODERATE
        else:
            urgency = UrgencyLevel.HIGH

        opportunity = None
        if daily_needed > 0:
            # Approximate 10 mins per 1000 steps
            mins_needed = max(
                GoalConfig.STEPS_MIN_GAP_MINUTES, int((daily_needed / 1000) * 10)
            )
            opportunity = pool.allocate(needed_mins=mins_needed)

        pct = int(pace_ratio * 100) if pace_ratio else 0
        k_today = round(steps_today_val / 1000, 1)
        k_goal = round(steps_goal_val / 1000, 1)
        k_needed = round(daily_needed / 1000, 1)

        if daily_needed <= 0:
            summary = f"Steps goal achieved: {k_today}k/{k_goal}k."
        else:
            summary = (
                f"Steps {k_today}k out of {k_goal}k goal. Needs {k_needed}k more today."
            )
            if opportunity:
                try:
                    dt = datetime.fromisoformat(
                        opportunity.start_time.replace("Z", "+00:00")
                    )
                    time_str = dt.strftime("%H:%M")
                except Exception:
                    time_str = opportunity.start_time
                summary += f" {opportunity.duration_mins}-min slot at {time_str}."

        return GoalSignal(
            goal_type="steps",
            domain="health",
            goal_label=f"Walk {steps.steps_goal} steps",
            urgency_level=urgency,
            pace_ratio=pace_ratio,
            days_remaining=days_remaining,
            daily_needed=float(daily_needed),
            calendar_opportunity=opportunity,
            insight_summary=summary,
        )

    @staticmethod
    def _process_daily_sleep(sleep, pool: AvailablePool) -> GoalSignal:

        # Read from summary + signals (single source of truth, no flat duplication).
        duration_sig = sleep.signals.sleep_duration
        debt_sig = sleep.signals.sleep_debt

        last_night_h = (
            duration_sig.metrics.get("last_night_hours") if duration_sig else None
        )
        goal_h = (
            duration_sig.metrics.get("target_hours") if duration_sig else None
        )
        debt_hours = (
            debt_sig.metrics.get("last_night_debt_h") if debt_sig else 0.0
        )
        if debt_hours is None:
            debt_hours = 0.0

        pace_ratio = (
            min(1.0, last_night_h / goal_h)
            if goal_h and last_night_h and goal_h > 0
            else 0.0
        )
        daily_needed = (
            max(0.0, goal_h - last_night_h)
            if goal_h and last_night_h
            else 0.0
        )
        days_remaining = 0

        t = GoalConfig.SLEEP_PACE_THRESHOLDS
        if pace_ratio >= t[0]:
            urgency = UrgencyLevel.ON_TRACK
        elif pace_ratio >= t[1]:
            urgency = UrgencyLevel.LOW
        elif pace_ratio >= t[2]:
            urgency = UrgencyLevel.MODERATE
        else:
            urgency = UrgencyLevel.HIGH

        opportunity = None
        if debt_hours >= GoalConfig.SLEEP_MIN_DEBT_FOR_NAP:
            opportunity = pool.allocate(needed_mins=GoalConfig.SLEEP_MIN_GAP_MINUTES)

        pct = int(pace_ratio * 100)
        ln = last_night_h
        g = goal_h
        if daily_needed <= 0.0:
            ln_str = f"{round(ln, 1)}" if ln is not None else "?"
            g_str = f"{round(g, 1)}" if g is not None else "?"
            summary = f"Sleep goal achieved: {ln_str}h/{g_str}h."
        else:
            ln_str = f"{round(ln, 1)}" if ln is not None else "?"
            g_str = f"{round(g, 1)}" if g is not None else "?"
            summary = f"Sleep {ln_str}h/{g_str}h ({pct}%), debt is {round(debt_hours, 1)}h."
            if opportunity:
                try:
                    dt = datetime.fromisoformat(
                        opportunity.start_time.replace("Z", "+00:00")
                    )
                    time_str = dt.strftime("%H:%M")
                except Exception:
                    time_str = opportunity.start_time
                summary += f" {opportunity.duration_mins}-min recovery slot available at {time_str}."

        return GoalSignal(
            goal_type="sleep",
            domain="health",
            goal_label=f"Sleep {goal_h} hours",
            urgency_level=urgency,
            pace_ratio=pace_ratio,
            days_remaining=days_remaining,
            daily_needed=daily_needed,
            calendar_opportunity=opportunity,
            insight_summary=summary,
        )

    @staticmethod
    def _get_days_remaining(goal_type: str, today: date) -> int:
        goal_type = goal_type.lower()
        if goal_type == "daily":
            return 0
        elif goal_type == "weekly":
            # 0 = Monday, 6 = Sunday
            return 6 - today.weekday()
        elif goal_type == "monthly":
            _, last_day = calendar.monthrange(today.year, today.month)
            return last_day - today.day
        elif goal_type == "yearly":
            end_of_year = date(today.year, 12, 31)
            return (end_of_year - today).days
        return 1

    @staticmethod
    def _process_custom_goal(
        user_goal: Dict[str, Any],
        pool: AvailablePool,
        health_signals: Optional[HealthSignalsBlock],
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> Optional[GoalSignal]:
        category = user_goal.get("category_code", "OVERALL")
        action = user_goal.get("action_code", "")

        # Deduplicate standard goals that are processed explicitly
        if action in ["EXERCISE_STEPS", "SLEEP_DURATION"]:
            return None

        goal_label = user_goal.get("goal_name", "Custom Goal")
        goal_type = user_goal.get("goal_type", "daily")

        # Calculate days remaining based on goal_type
        today = date.today()
        days_rem = GoalProgressProcessor._get_days_remaining(goal_type, today)
        if days_rem is None or days_rem < 0:
            days_rem = 1

        raw_target = user_goal.get("target_value")
        target_val = float(raw_target) if raw_target is not None else 0.0
        # Avoid division by zero, if days_rem is 0 (today is the last day), treat as 1 day to distribute load
        try:
            daily_needed = (
                target_val / max(1, days_rem) if goal_type != "daily" else target_val
            )
        except TypeError:
            daily_needed = 0.0

        pace = 0.0

        urgency = UrgencyLevel.LOW
        summary = f"{goal_label} - Missing tracking data."
        opportunity = None

        if user_profile:
            if action == "WEIGHT_LOSS":
                current_weight = user_profile.get("weight_kg")
                bmi = user_profile.get("bmi")
                if current_weight:
                    summary = f"Goal: {goal_label}. Current weight: {current_weight}kg (BMI: {bmi}). Stay consistent with diet and exercise."

            elif action == "BREAK_INTERVAL":
                summary = f"Goal: {goal_label}. Consider scheduling short breaks between meetings."
                opportunity = pool.allocate(needed_mins=5)
                if opportunity:
                    try:
                        from datetime import datetime

                        dt = datetime.fromisoformat(
                            opportunity.start_time.replace("Z", "+00:00")
                        )
                        time_str = dt.strftime("%H:%M")
                    except Exception:
                        time_str = opportunity.start_time
                    summary += f" Recommended 5-min break slot available at {time_str}."

        start_date_str = None
        end_date_str = None

        if goal_type == "weekly":
            start_date_str = (today - timedelta(days=today.weekday())).isoformat()
            end_date_str = (today + timedelta(days=days_rem)).isoformat()
        elif goal_type == "monthly":
            start_date_str = date(today.year, today.month, 1).isoformat()
            end_date_str = (today + timedelta(days=days_rem)).isoformat()
        elif goal_type == "yearly":
            start_date_str = date(today.year, 1, 1).isoformat()
            end_date_str = (today + timedelta(days=days_rem)).isoformat()
        else:  # daily
            start_date_str = today.isoformat()
            end_date_str = today.isoformat()

        return GoalSignal(
            goal_type="custom",
            domain=category.lower(),
            goal_label=goal_label,
            urgency_level=urgency,
            pace_ratio=pace,
            days_remaining=days_rem,
            daily_needed=daily_needed,
            calendar_opportunity=opportunity,
            insight_summary=summary,
            start_date=start_date_str,
            end_date=end_date_str,
        )

"""Insight Service Implementation"""

import asyncio
import json
import random
import re
from calendar import monthrange
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from agents.llm_helper import (
    llm_response_text,
    parse_json_object_from_llm_text,
    resolve_generation_language,
    translate_insight_dict,
)
from agents.llm_manager import LLMManager
from agents.prompt import get_cross_module_rewrite_prompts
from agents.prompt_day_context import get_day_context_prompt
from agents.prompt_health_insight import get_health_insight_prompt
from agents.prompt_overall_insight import get_overall_insight_prompt
from agents.prompt_rule_based import (  # Great Job prompts; Need Attention prompts; Opportunity prompts
    get_active_window_insight_prompt,
    get_bedtime_window_insight_prompt,
    get_calendar_pattern_insight_prompt,
    get_contextual_suggestion_insight_prompt,
    get_evening_flexible_insight_prompt,
    get_health_goal_insight_prompt,
    get_morning_start_insight_prompt,
    get_recovery_break_insight_prompt,
    get_safety_risk_insight_prompt,
    get_task_type_insight_prompt,
    get_weekend_lifestyle_insight_prompt,
    get_wind_down_window_insight_prompt,
)
from clients.redis_client import RedisClient
from langchain_core.messages import HumanMessage, SystemMessage
from models.models import BalanceScoreDTO, MoodDTO
from services.executor.cache_helpers import CacheHelpers
from services.executor.constant import (
    APIResponseKeys,
    CalendarDataConstants,
    FinanceDataConstants,
    HealthDataConstants,
    InsightGroupConstants,
    TargetKeys,
    TimeDataConstants,
)
from services.executor.prepare_calendar_data import PrepareCalendarData
from services.executor.prepare_health_data import PrepareHealthData
from services.executor.prepare_time_data import PrepareTimeData
from services.external_api_service import IExternalAPIService
from services.insight_service import InsightService
from utils.logger import logger


class InsightServiceImpl(InsightService):
    """Implementation of insight service with Bedrock LLM"""

    def __init__(self):
        # Initialize Bedrock model using LLMManager
        llm_manager = LLMManager(provider="bedrock")
        self.llm = llm_manager.create_model(temperature=0.1)

        # External API service will be injected via setter
        self.external_api_service: Optional[IExternalAPIService] = None

        # Redis client for caching insights
        self.redis_client = RedisClient()

        logger.info("✅ InsightService initialized with Bedrock LLM")

    def set_external_api_service(self, external_api_service: IExternalAPIService):
        """Set external API service instance"""
        self.external_api_service = external_api_service

    def _format_calendar_data_for_llm(
        self, events: List[Dict[str, Any]], timezone: Optional[str] = None
    ) -> str:
        if not events:
            return "No calendar events found for today."

        formatted = []
        formatted.append(f"Today's Calendar Events ({len(events)} total):\n")

        for idx, event in enumerate(events, 1):
            summary = event.get("summary", "")
            start_time = event.get("startTime", "")
            end_time = event.get("endTime", "")
            location = event.get("location", "")
            description = event.get("description", "")
            all_day = event.get("allDay", False)
            participants = event.get("participants", [])
            # Convert times to target timezone if provided
            if timezone and start_time and end_time:
                try:
                    # Parse start_time and end_time (assuming ISO format)
                    start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                    end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))

                    # Convert to target timezone
                    target_tz = ZoneInfo(timezone)
                    start_dt = start_dt.astimezone(target_tz)
                    end_dt = end_dt.astimezone(target_tz)

                    # Format as readable time string
                    start_time = start_dt.strftime("%Y-%m-%d %H:%M:%S")
                    end_time = end_dt.strftime("%Y-%m-%d %H:%M:%S")
                except Exception as e:
                    logger.warning(
                        f"⚠️ Failed to convert timezone for event {idx}: {str(e)}"
                    )
                    # Keep original times if conversion fails

            event_str = f"\n{idx}. {summary}\n"
            if all_day:
                event_str += "   Type: All-day event\n"
            else:
                event_str += f"   Time: {start_time} - {end_time}\n"
            if location:
                event_str += f"   Location: {location}\n"
            if participants:
                # Format participants: name (email)
                formatted_participants = []
                for participant in participants[:3]:  # Take first 3 participants
                    if isinstance(participant, dict):
                        name = participant.get("name", "")
                        email = participant.get("email", "")
                        if name and email:
                            formatted_participants.append(f"{name} ({email})")
                        elif email:
                            formatted_participants.append(email)
                        elif name:
                            formatted_participants.append(name)
                    elif isinstance(participant, str):
                        formatted_participants.append(participant)

                participants_str = ", ".join(formatted_participants)

                # If more than 3 participants (>= 4), add "and X more"
                if len(participants) > 3:
                    remaining_count = len(participants) - 3
                    if remaining_count > 0:
                        participants_str += f", and {remaining_count} more"

                event_str += f"   Participants: {participants_str}\n"
            if description:
                event_str += f"   Description: {description}\n"

            formatted.append(event_str)

        return "".join(formatted)

    def _parse_json_response(self, content: str) -> dict:
        raw = (content or "").strip()
        parsed = parse_json_object_from_llm_text(raw)

        if not isinstance(parsed, dict):
            logger.warning("⚠️ Failed to parse JSON from LLM response. Using fallback.")
            return {
                "point1": raw if raw else "Could not parse response.",
                "point2": "Please try again or contact support.",
            }

        if "point1" not in parsed:
            parsed["point1"] = "No summary provided."
        if "point2" not in parsed:
            parsed["point2"] = "No recommendation provided."

        logger.info("✅ Successfully parsed JSON response from LLM")
        return parsed

    _OVERALL_INSIGHT_KEYS = ("great_job", "need_attention", "opportunity")

    _EVENING_OVERALL_INSIGHT_GROUPS = frozenset(
        {
            "wind_down_window",
            "bedtime_window",
            "evening_flexible",
            "evening_after_work",
        }
    )

    @staticmethod
    def _compact_overall_insight(insight: Dict[str, Any]) -> Dict[str, str]:
        """Return only non-empty overall insight categories."""
        if not isinstance(insight, dict):
            return {}
        compact: Dict[str, str] = {}
        for key in InsightServiceImpl._OVERALL_INSIGHT_KEYS:
            value = insight.get(key)
            if isinstance(value, str) and value.strip():
                compact[key] = value.strip()
        return compact

    def _normalize_wind_down_terms(
        self, text: str, language: Optional[str] = "en-US"
    ) -> str:
        """Normalize wind-down wording for Vietnamese outputs."""
        if not isinstance(text, str) or not text:
            return text

        # Only force normalization for Vietnamese responses
        language_code = (language or "").lower()
        if not language_code.startswith("vi"):
            return text

        normalized = text
        dash_pattern = r"(?:[-\u2010\u2011\u2012\u2013\u2014]|\s)"

        normalized = re.sub(
            rf"\b([Gg]iai đoạn)\s+wind{dash_pattern}down\b",
            lambda m: (
                "Giai đoạn thư giãn"
                if m.group(1).startswith("G")
                else "giai đoạn thư giãn"
            ),
            normalized,
        )
        normalized = re.sub(
            rf"\b([Tt]hời gian)\s+wind{dash_pattern}down\b",
            lambda m: (
                "Thời gian thư giãn"
                if m.group(1).startswith("T")
                else "thời gian thư giãn"
            ),
            normalized,
        )

        # Normalize remaining standalone forms
        normalized = re.sub(
            rf"\bWind{dash_pattern}down\b", "Giai đoạn thư giãn", normalized
        )
        normalized = re.sub(
            rf"\bwind{dash_pattern}down\b", "giai đoạn thư giãn", normalized
        )

        return normalized

    def _clean_json_response(self, response_text: str) -> str:
        # Remove markdown code block delimiters
        cleaned = response_text.strip()

        # Remove ```json or ``` from start
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]

        # Remove ``` from end
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        return cleaned.strip()

    def _get_cache_key(
        self, user_id: str, language: str, insight_type: str = "productivity"
    ) -> str:
        return f"{insight_type}_insight:{user_id}:{language}"

    def _extract_fresh_mood(
        self,
        latest_mood: Optional[Dict[str, Any]],
        time_data: Optional[Dict[str, Any]],
    ) -> tuple[Optional[str], bool]:
        try:
            if not latest_mood:
                return None, True

            mood_value = latest_mood.get("mood")
            mood_date_raw = latest_mood.get("moodDate") or latest_mood.get("mood_date")
            if not mood_value or not mood_date_raw:
                return None, True

            # Parse mood_date (typically ISO UTC like "2026-05-16T17:13:27.824Z")
            iso_input = str(mood_date_raw).replace("Z", "+00:00")
            try:
                mood_dt = datetime.fromisoformat(iso_input)
            except ValueError:
                logger.warning(
                    f"⚠️ Unparseable moodDate '{mood_date_raw}', treating mood as not-today"
                )
                return None, True

            if mood_dt.tzinfo is None:
                mood_dt = mood_dt.replace(tzinfo=ZoneInfo("UTC"))

            tz = (time_data or {}).get(TimeDataConstants.KEY_TZ)
            current_date = (time_data or {}).get(TimeDataConstants.KEY_CURRENT_DATE)
            if tz is None or current_date is None:
                # Without a reference local date we can't decide freshness;
                # err on the safe side and treat as not-today.
                return None, True

            mood_local_date = mood_dt.astimezone(tz).date()
            if mood_local_date == current_date:
                return mood_value, False
            return None, True
        except Exception as e:
            logger.warning(f"⚠️ Failed to evaluate mood freshness: {str(e)}")
            return None, True

    async def _fetch_today_reminders(
        self,
        user_id: str,
        timezone: Optional[str],
        current_dt: datetime,
    ) -> List[Dict[str, Any]]:
        if not self.external_api_service:
            return []
        try:
            local_today = current_dt.date().isoformat()
            raw = await self.external_api_service.get_reminders(
                user_id=user_id,
                start_date=local_today,
                end_date=local_today,
                timezone=timezone,
                page=1,
                size=50,
            )
            if not raw:
                return []

            now_utc = datetime.now(ZoneInfo("UTC"))
            result: List[Dict[str, Any]] = []
            for item in raw:
                if not isinstance(item, dict):
                    continue
                due_raw = item.get("dueDate") or ""
                completed = bool(item.get("completed", False))
                is_overdue = False
                if not completed and due_raw:
                    try:
                        due_dt = datetime.fromisoformat(
                            str(due_raw).replace("Z", "+00:00")
                        )
                        if due_dt.tzinfo is None:
                            due_dt = due_dt.replace(tzinfo=ZoneInfo("UTC"))
                        is_overdue = due_dt < now_utc
                    except ValueError:
                        logger.warning(
                            f"⚠️ Unparseable reminder dueDate '{due_raw}', "
                            f"skipping overdue check"
                        )
                result.append(
                    {
                        "title": item.get("title") or "",
                        "notes": item.get("notes") or "",
                        "dueDate": due_raw,
                        "status": item.get("status") or "",
                        "completed": completed,
                        "is_overdue": is_overdue,
                    }
                )
            logger.info(
                f"📝 Reminders today for user {user_id}: total={len(result)}, "
                f"overdue={sum(1 for r in result if r['is_overdue'])}, "
                f"completed={sum(1 for r in result if r['completed'])}"
            )
            return result
        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch reminders for user {user_id}: {str(e)}")
            return []

    def _merge_calendar_with_reminders(
        self,
        calendar_data: str,
        reminders_section: str,
    ) -> str:
        if not reminders_section:
            return calendar_data
        return f"{calendar_data}\n\n{reminders_section}"

    def _build_reminders_section(
        self,
        reminders: List[Dict[str, Any]],
        timezone: Optional[str],
    ) -> str:
        if not reminders:
            return ""

        total = len(reminders)
        completed_count = sum(1 for r in reminders if r.get("completed"))
        overdue_count = sum(1 for r in reminders if r.get("is_overdue"))
        pending_count = total - completed_count

        try:
            tz = ZoneInfo(timezone) if timezone else ZoneInfo("UTC")
        except Exception:
            tz = ZoneInfo("UTC")

        lines: List[str] = []
        lines.append(
            f"**Today's Reminders ({total} total | "
            f"{pending_count} pending | {overdue_count} overdue | "
            f"{completed_count} completed):**"
        )

        for idx, r in enumerate(reminders, 1):
            title = r.get("title") or "(untitled)"
            notes = (r.get("notes") or "").strip()
            status = r.get("status") or (
                "completed" if r.get("completed") else "needsAction"
            )
            due_raw = r.get("dueDate") or ""
            time_label = ""
            if due_raw:
                try:
                    due_dt = datetime.fromisoformat(str(due_raw).replace("Z", "+00:00"))
                    if due_dt.tzinfo is None:
                        due_dt = due_dt.replace(tzinfo=ZoneInfo("UTC"))
                    time_label = due_dt.astimezone(tz).strftime("%H:%M")
                except ValueError:
                    time_label = ""
            tag = "[OVERDUE] " if r.get("is_overdue") else ""
            due_part = f" (due {time_label})" if time_label else ""
            line = f"{idx}. {tag}{title}{due_part} — {status}"
            if notes:
                line += f"\n   notes: {notes}"
            lines.append(line)

        lines.append("")
        lines.append(
            "**Reminder data notes (use with TODAY'S REMINDERS AWARENESS in system prompt):**"
        )
        lines.append("- Today only — do not infer tasks from other days.")
        lines.append(
            "- `[OVERDUE]` = not completed and past due time; weigh with meeting load and free slots above."
        )
        lines.append(
            '- If citing a reminder by name, use its exact title in "quotes" (do not translate the title).'
        )
        lines.append(
            "- Productivity (point1/point2) or overall (great_job / need_attention / opportunity): "
            "mention reminders only when they materially change the suggestion."
        )

        return "\n".join(lines)

    def _extract_user_targets(
        self, goals: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Extract user targets from goals array based on action_code mapping.

        Args:
            goals: List of goal dictionaries with action_code, target_value, unit

        Returns:
            Dictionary mapping target_field to {"value": target_value, "unit": unit}
        """
        # Mapping from action_code to target_field
        action_code_to_target = {
            "EXERCISE_STEPS": "target_steps_per_day",
            "HEART_RATE": "target_heart_rate_bpm",
            "WATER_INTAKE": "target_water_intake_liters",
            "SLEEP_DURATION": "target_sleep_hours",
            "BREAK_INTERVAL": "break_interval_hours",
            "BREAK_DURATION": "break_duration_min",
            "DAILY_TASKS": "target_tasks_per_day",
            "ACTIVE_HOURS": "target_active_hours",
        }

        targets_data = {}

        for goal in goals:
            if not isinstance(goal, dict):
                continue

            action_code = goal.get("action_code")
            if not action_code or action_code not in action_code_to_target:
                continue

            # Only process active goals
            if not goal.get("active", False):
                continue

            target_field = action_code_to_target[action_code]
            target_value = goal.get("target_value")
            unit = goal.get("unit", "")

            # Only add if target_value exists
            if target_value is not None:
                targets_data[target_field] = {
                    "value": target_value,
                    "unit": unit,
                }

        return targets_data

    def _safe_float(self, value: Any) -> Optional[float]:
        """Safely convert to float, return None when invalid."""
        try:
            if value is None or value == "":
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _safe_score_ratio(self, value: Any) -> Optional[float]:
        """Normalize score to ratio [0..1] when possible.

        If input looks like percent (>1), convert by dividing by 100.
        """
        v = self._safe_float(value)
        if v is None:
            return None
        if v > 1:
            return v / 100.0
        return v

    def _is_evening_window_for_cross_module(self, current_dt: datetime) -> bool:
        """Cross-module replacement window: 18:00 <= now < 24:00."""
        return 18 <= current_dt.hour < 24

    def _should_random_replace_need_attention(self) -> bool:
        """50/50 random gate for replacement."""
        return random.random() < 0.5

    def _is_end_of_month_window(self, ym: str, today_local: datetime) -> bool:
        """True on end-of-month and 1st/2nd of following month (same as finance)."""
        try:
            dt = datetime.strptime(ym, "%Y-%m").date()
        except ValueError:
            return False
        today = today_local.date()
        last_day = monthrange(dt.year, dt.month)[1]
        end_date = datetime(dt.year, dt.month, last_day).date()
        next_m = dt.month + 1
        next_y = dt.year
        if next_m > 12:
            next_m = 1
            next_y += 1
        follow_day2 = datetime(next_y, next_m, 2).date()
        return end_date <= today <= follow_day2

    def _prev_months(self, ym: str, n: int) -> List[str]:
        """Return list of n prior months as YYYY-MM in ascending order."""
        dt = datetime.strptime(ym, "%Y-%m")
        result = []
        for i in range(n, 0, -1):
            m = dt.month - i
            y = dt.year
            while m <= 0:
                m += 12
                y -= 1
            result.append(f"{y}-{m:02d}")
        return result

    def _month_start_end(self, ym: str) -> tuple[str, str]:
        """Return YYYY-MM-DD start/end dates for a month string YYYY-MM."""
        dt = datetime.strptime(ym, "%Y-%m")
        last_day = monthrange(dt.year, dt.month)[1]
        start = datetime(dt.year, dt.month, 1).date().isoformat()
        end = datetime(dt.year, dt.month, last_day).date().isoformat()
        return start, end

    def _eom_reference_months(self, ym: str, current_dt: datetime) -> tuple[str, str]:
        """Resolve comparison months for EOM rules.

        - End of month day: current=ym, prev=ym-1
        - Day 1/2 of next month: current=ym-1, prev=ym-2
        """
        if current_dt.day in (1, 2):
            prev_months = self._prev_months(ym, 2)
            if len(prev_months) == 2:
                return prev_months[1], prev_months[0]
        prev1 = self._prev_months(ym, 1)
        prev_month = prev1[0] if prev1 else ym
        return ym, prev_month

    def _avg_balance_metric_from_rows(
        self, rows: List[BalanceScoreDTO], metric_field: str
    ) -> Optional[float]:
        """Average a positive balance metric from rows.

        Only values > 0 are counted. Zero/negative/missing values are ignored.
        If no valid values remain, return None.
        """
        vals: List[float] = []
        for row in rows:
            p = self._safe_float(getattr(row, metric_field, None))
            if p is not None and p > 0:
                vals.append(p)
        if not vals:
            return None
        avg = sum(vals) / len(vals)
        return avg

    def _sum_productivity_total_hours(
        self, rows: List[Dict[str, Any]]
    ) -> Optional[float]:
        """Sum productivity totalHours from daily rows.

        - Uses field `totalHours` as daily work-hour value.
        - Ignores invalid/missing/negative values.
        - Returns None when no valid daily value is available.
        """
        vals: List[float] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            h = self._safe_float(row.get("totalHours"))
            if h is not None and h >= 0:
                vals.append(h)
        if not vals:
            return None
        return sum(vals)

    def _extract_health_score_current(
        self,
        health_data: Optional[Dict[str, Any]],
        health_params: Optional[Dict[str, Any]],
    ) -> Optional[float]:
        """Extract current health score from available sources."""
        # 1) Try health summaries (if available)
        if isinstance(health_data, dict):
            scores = []
            for summary in health_data.values():
                if hasattr(summary, "data"):
                    summary_data = getattr(summary, "data", {}) or {}
                elif isinstance(summary, dict):
                    summary_data = summary.get("data", {}) or {}
                else:
                    summary_data = {}
                score = self._safe_score_ratio(summary_data.get("healthScore"))
                if score is not None:
                    scores.append(score)
            if scores:
                return sum(scores) / len(scores)

        # 2) Fallback from weekly health progress (progress_percentage)
        if isinstance(health_params, dict):
            weekly = health_params.get(
                HealthDataConstants.KEY_WEEKLY_HEALTH_PROGRESS, {}
            )
            if isinstance(weekly, dict):
                p = self._safe_score_ratio(
                    weekly.get(HealthDataConstants.KEY_PROGRESS_PERCENTAGE)
                )
                if p is not None:
                    return p
        return None

    async def _collect_cross_module_inputs(
        self,
        user_id: str,
        timezone: Optional[str],
        current_dt: datetime,
        health_data: Optional[Dict[str, Any]],
        health_params: Optional[Dict[str, Any]],
        calendar_metrics: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Collect cross-module signals used by rule evaluation."""
        ym = current_dt.strftime("%Y-%m")
        # EOM rules run on: month end, and day 1/day 2 of following month.
        # With ym=current month string, explicit day 1/2 check keeps this window active.
        in_eom_window = self._is_end_of_month_window(
            ym, current_dt
        ) or current_dt.day in (
            1,
            2,
        )
        inputs: Dict[str, Any] = {
            "work_hours_week": None,
            "sleep_score": None,
            "health_score": None,
            "health_score_current": None,
            "health_score_prev": None,
            "productivity_score": None,
            "productivity_score_current": None,
            "productivity_score_prev": None,
            "work_hours_current_month": None,
            "work_hours_prev_month": None,
            "avg_productivity_score_last_3_months": None,
            "income_current_month": None,
            "income_prev_month": None,
            "income_avg": None,
            "income_growth_rate_last_3_months": None,
            "income_growth_rate": None,
            "income_variance": None,
            "savings_rate": None,
            "finance_score": None,
            "ym": ym,
            "in_eom_window": in_eom_window,
        }

        # Reuse available local context first
        if isinstance(calendar_metrics, dict):
            work_hours_week = self._safe_float(
                calendar_metrics.get(CalendarDataConstants.KEY_WORK_EVENTS_HOURS_WEEK)
            )
            inputs["work_hours_week"] = work_hours_week

        if isinstance(health_params, dict):
            sleep_q = health_params.get(HealthDataConstants.KEY_SLEEP_QUALITY_SCORE)
            inputs["sleep_score"] = self._safe_score_ratio(sleep_q)

        if not self.external_api_service:
            return inputs

        # ===== Part 1: Any day data =====
        # Always collect baseline data for rules that can trigger any day.
        prev_months_3 = self._prev_months(ym, 3)
        # Fetch exactly 4 months for reuse: [ym-3, ym-2, ym-1, ym].
        # On day 1/2, this still includes the EOM reference set (ym-3, ym-2, ym-1).
        months_to_fetch = prev_months_3 + [ym]
        finance_results = await asyncio.gather(
            *[
                self.external_api_service.get_finance_summary(user_id=user_id, month=m)
                for m in months_to_fetch
            ],
            return_exceptions=True,
        )
        finance_by_month: Dict[str, Dict[str, Any]] = {}
        for m, res in zip(months_to_fetch, finance_results):
            if isinstance(res, dict):
                finance_by_month[m] = res

        cur = finance_by_month.get(ym, {})
        prev_m = prev_months_3[-1] if prev_months_3 else None
        prev = finance_by_month.get(prev_m, {}) if prev_m else {}

        incomes_last_3: List[float] = []
        productivity_last_3: List[float] = []
        for m in prev_months_3:
            item = finance_by_month.get(m, {})
            inc = self._safe_float(item.get("totalIncome"))
            if inc is not None:
                incomes_last_3.append(inc)
            p = self._safe_score_ratio(item.get("productivityScore"))
            if p is not None:
                productivity_last_3.append(p)

        inputs["income_current_month"] = self._safe_float(cur.get("totalIncome"))
        inputs["income_avg"] = (
            sum(incomes_last_3) / len(incomes_last_3) if incomes_last_3 else None
        )

        inputs["savings_rate"] = self._safe_score_ratio(
            cur.get("savingsRatePercentage")
        )
        # Prefer balance/score for Any-day metrics (productivity/finance, optional health)
        balance_date = current_dt.date().isoformat()
        balance_payload: Optional[BalanceScoreDTO] = None
        try:
            balance_payload = await self.external_api_service.get_balance_score(
                user_id=user_id,
                date=balance_date,
                timezone=timezone,
            )
        except Exception:
            balance_payload = None

        if isinstance(balance_payload, BalanceScoreDTO):
            inputs["productivity_score"] = self._safe_float(
                balance_payload.productivityScore
            )
            inputs["finance_score"] = self._safe_float(balance_payload.financeScore)
            # Some responses may include healthScore; if present, prefer it.
            health_from_balance = self._safe_float(balance_payload.healthScore)
            if health_from_balance is not None:
                inputs["health_score"] = health_from_balance
        else:
            inputs["finance_score"] = self._safe_score_ratio(cur.get("financeScore"))

        # ===== Part 2: End-of-month data =====
        # Only enrich data for rules in "Once - End of Month + 1st+2nd".
        if not in_eom_window:
            return inputs

        current_ref_ym, prev_ref_ym = self._eom_reference_months(ym, current_dt)
        prev_prev_ref = self._prev_months(prev_ref_ym, 1)
        prev_prev_ref_ym = prev_prev_ref[0] if prev_prev_ref else prev_ref_ym
        income_current_ref = self._safe_float(
            finance_by_month.get(current_ref_ym, {}).get("totalIncome")
        )
        income_prev_ref = self._safe_float(
            finance_by_month.get(prev_ref_ym, {}).get("totalIncome")
        )
        income_prev_prev_ref = self._safe_float(
            finance_by_month.get(prev_prev_ref_ym, {}).get("totalIncome")
        )
        inputs["income_prev_month"] = income_prev_ref
        cur_start, cur_end = self._month_start_end(current_ref_ym)
        prev_start, prev_end = self._month_start_end(prev_ref_ym)
        prev_prev_start, prev_prev_end = self._month_start_end(prev_prev_ref_ym)
        cur_rows = await self.external_api_service.get_balance_scores_by_range(
            user_id=user_id,
            start_date=cur_start,
            end_date=cur_end,
            timezone=timezone,
        )
        prev_rows = await self.external_api_service.get_balance_scores_by_range(
            user_id=user_id,
            start_date=prev_start,
            end_date=prev_end,
            timezone=timezone,
        )
        prev_prev_rows = await self.external_api_service.get_balance_scores_by_range(
            user_id=user_id,
            start_date=prev_prev_start,
            end_date=prev_prev_end,
            timezone=timezone,
        )
        productivity_current = self._avg_balance_metric_from_rows(
            cur_rows, APIResponseKeys.PRODUCTIVITY_SCORE
        )
        productivity_prev = self._avg_balance_metric_from_rows(
            prev_rows, APIResponseKeys.PRODUCTIVITY_SCORE
        )
        productivity_prev_prev = self._avg_balance_metric_from_rows(
            prev_prev_rows, APIResponseKeys.PRODUCTIVITY_SCORE
        )
        health_current = self._avg_balance_metric_from_rows(
            cur_rows, APIResponseKeys.HEALTH_SCORE
        )
        health_prev = self._avg_balance_metric_from_rows(
            prev_rows, APIResponseKeys.HEALTH_SCORE
        )

        inputs["productivity_score_current"] = productivity_current
        inputs["productivity_score_prev"] = productivity_prev
        inputs["health_score_current"] = health_current
        inputs["health_score_prev"] = health_prev

        # Strict rule: avg only when all 3 months are present.
        prod_last_3 = (productivity_current, productivity_prev, productivity_prev_prev)
        inputs["avg_productivity_score_last_3_months"] = (
            sum(prod_last_3) / 3 if all(x is not None for x in prod_last_3) else None
        )
        cur_prod_rows, prev_prod_rows = await asyncio.gather(
            self.external_api_service.get_productivity_summaries_by_range(
                user_id=user_id,
                start_date=cur_start,
                end_date=cur_end,
                timezone=timezone,
            ),
            self.external_api_service.get_productivity_summaries_by_range(
                user_id=user_id,
                start_date=prev_start,
                end_date=prev_end,
                timezone=timezone,
            ),
            return_exceptions=True,
        )
        if isinstance(cur_prod_rows, list):
            inputs["work_hours_current_month"] = self._sum_productivity_total_hours(
                cur_prod_rows
            )
        if isinstance(prev_prod_rows, list):
            inputs["work_hours_prev_month"] = self._sum_productivity_total_hours(
                prev_prod_rows
            )

        income_last_3_ref = (income_prev_prev_ref, income_prev_ref, income_current_ref)
        if all(x is not None for x in income_last_3_ref) and income_prev_prev_ref > 0:
            inputs["income_growth_rate_last_3_months"] = (
                income_current_ref - income_prev_prev_ref
            ) / income_prev_prev_ref

        if (
            income_prev_ref is not None
            and income_current_ref is not None
            and income_prev_ref > 0
        ):
            inputs["income_growth_rate"] = (
                income_current_ref - income_prev_ref
            ) / income_prev_ref

        if all(x is not None for x in income_last_3_ref):
            avg = sum(income_last_3_ref) / 3
            if avg > 0:
                variance = sum((x - avg) ** 2 for x in income_last_3_ref) / 3
                std = variance**0.5
                inputs["income_variance"] = std / avg

        return inputs

    async def _rewrite_cross_module_sections_with_llm(
        self, section_texts: Dict[str, List[str]], language: str = "en-US"
    ) -> Dict[str, str]:
        """Use LLM to merge/rewrite section insights in target language."""
        great_job_items = section_texts.get("great_job", [])[:3]
        need_attention_items = section_texts.get("need_attention", [])[:3]

        # Fast return when there is nothing to rewrite.
        if not great_job_items and not need_attention_items:
            return {"great_job": "", "need_attention": ""}

        payload = {
            "language": language,
            "great_job_items": great_job_items,
            "need_attention_items": need_attention_items,
        }

        gen_language = resolve_generation_language(language)
        system_prompt, user_prompt = get_cross_module_rewrite_prompts(
            language=gen_language,
            input_json=json.dumps(payload, ensure_ascii=False),
        )

        try:

            response = await self.llm.ainvoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )
            content_text = llm_response_text(response)
            parsed = self._parse_json_response(content_text)
            if isinstance(parsed, dict):
                result = {
                    "great_job": str(parsed.get("great_job", "") or ""),
                    "need_attention": str(parsed.get("need_attention", "") or ""),
                }
                return await translate_insight_dict(
                    result,
                    language,
                    self.llm,
                    insight_type="overall_insight",
                )
        except Exception as e:
            logger.warning(
                f"⚠️ Failed to rewrite cross-module sections with LLM: {str(e)}"
            )

        # Fallback to deterministic formatter if LLM fails.
        return {
            "great_job": self._format_cross_module_insights(great_job_items),
            "need_attention": self._format_cross_module_insights(need_attention_items),
        }

    async def _evaluate_cross_module_rules(
        self,
        inputs: Dict[str, Any],
        current_dt: datetime,
        language: str = "en-US",
    ) -> Dict[str, str]:
        """Evaluate rules then return rewritten per-section insights."""
        matches: List[Dict[str, str]] = []
        in_eom_window = bool(inputs.get("in_eom_window"))

        income_cur = inputs.get("income_current_month")
        income_prev = inputs.get("income_prev_month")
        income_avg = inputs.get("income_avg")
        income_growth_3m = inputs.get("income_growth_rate_last_3_months")
        income_growth_rate = inputs.get("income_growth_rate")
        income_variance = inputs.get("income_variance")
        productivity_cur = inputs.get("productivity_score_current")
        productivity_prev = inputs.get("productivity_score_prev")
        productivity_avg_3m = inputs.get("avg_productivity_score_last_3_months")
        health_cur = inputs.get("health_score_current")
        health_prev = inputs.get("health_score_prev")
        sleep_score = inputs.get("sleep_score")
        work_hours_week = inputs.get("work_hours_week")
        savings_rate = inputs.get("savings_rate")
        finance_score = inputs.get("finance_score")
        productivity_score = inputs.get("productivity_score")
        health_score = inputs.get("health_score")
        work_hours_current_month = inputs.get("work_hours_current_month")
        work_hours_prev_month = inputs.get("work_hours_prev_month")

        logger.info(f"🧠 Cross-Module Intelligence inputs: {inputs}")

        # Thresholds
        productivity_threshold = CalendarDataConstants.PRODUCTIVITY_THRESHOLD
        healthy_sleep_threshold = HealthDataConstants.SLEEP_QUALITY_GOOD
        health_threshold = HealthDataConstants.HEALTH_THRESHOLD
        sleep_threshold = HealthDataConstants.SLEEP_QUALITY_THRESHOLD
        work_hour_threshold = CalendarDataConstants.WORK_LOAD_HIGH_THRESHOLD_HOURS
        finance_threshold = FinanceDataConstants.KEY_FINANCE_THRESHOLD
        savings_rate_threshold = FinanceDataConstants.KEY_SAVINGS_RATE_THRESHOLD

        # 1) Income-Productivity: work hours increase but income not increasing
        # NOTE: work_hours_current_month may be unavailable; skip safely.
        if in_eom_window:
            work_hours_current_month = inputs.get("work_hours_current_month")
            work_hours_prev_month = inputs.get("work_hours_prev_month")
            if (
                work_hours_current_month is not None
                and work_hours_prev_month is not None
                and income_cur is not None
                and income_prev is not None
                and work_hours_current_month > work_hours_prev_month * 1.20
                and income_cur <= income_prev
            ):
                matches.append(
                    {
                        "section": "need_attention",
                        "text": "You worked significantly more hours this month but your income did not increase.",
                    }
                )

        # 2) Productivity improves and income increases
        if (
            in_eom_window
            and productivity_cur is not None
            and productivity_prev is not None
            and income_cur is not None
            and income_prev is not None
            and productivity_cur > productivity_prev
            and income_cur > income_prev
        ):
            matches.append(
                {
                    "section": "great_job",
                    "text": "Your productivity improvement appears to be contributing to income growth.",
                }
            )

        # 3) High productivity but income stagnant for 3 months
        if (
            in_eom_window
            and productivity_avg_3m is not None
            and income_growth_3m is not None
            and productivity_avg_3m > productivity_threshold
            and income_growth_3m < 0.05
        ):
            matches.append(
                {
                    "section": "need_attention",
                    "text": "Your productivity has been strong, but income has remained flat for three months.",
                }
            )

        # 4) Healthy work-life balance maintained while income stable/improving
        if (
            in_eom_window
            and work_hours_week is not None
            and sleep_score is not None
            and income_growth_3m is not None
            and work_hours_week <= work_hour_threshold
            and sleep_score >= healthy_sleep_threshold
            and income_growth_3m >= 0
        ):
            matches.append(
                {
                    "section": "great_job",
                    "text": "You are maintaining a healthy work-life balance while sustaining income.",
                }
            )

        # 5) Life efficiency: productivity + health + savings
        if (
            productivity_score is not None
            and health_score is not None
            and savings_rate is not None
            and productivity_score > productivity_threshold
            and health_score > health_threshold
            and savings_rate > savings_rate_threshold
        ):
            matches.append(
                {
                    "section": "great_job",
                    "text": "You are maintaining a strong balance between productivity, health, and finances.",
                }
            )

        # 6) Life efficiency: productivity + health + finance score
        if (
            productivity_score is not None
            and health_score is not None
            and finance_score is not None
            and productivity_score > productivity_threshold
            and health_score > health_threshold
            and finance_score > finance_threshold
        ):
            matches.append(
                {
                    "section": "great_job",
                    "text": "You are maintaining a strong balance between productivity, health, and finances.",
                }
            )

        # 7) High income but declining health
        if (
            in_eom_window
            and income_cur is not None
            and income_avg is not None
            and health_cur is not None
            and income_cur > income_avg
            and health_cur < health_threshold
        ):
            matches.append(
                {
                    "section": "need_attention",
                    "text": "Your income is strong, but declining health indicators may signal overwork.",
                }
            )

        # 8) Strong health but low productivity impacts income growth
        if (
            in_eom_window
            and health_cur is not None
            and productivity_cur is not None
            and income_growth_rate is not None
            and health_cur > health_threshold
            and productivity_cur < productivity_threshold
            and income_growth_rate < 0.05
        ):
            matches.append(
                {
                    "section": "need_attention",
                    "text": "Your health indicators are strong, but productivity may be limiting income growth.",
                }
            )

        # 9) Sustainable performance risk
        if (
            in_eom_window
            and productivity_cur is not None
            and income_growth_rate is not None
            and health_cur is not None
            and health_prev is not None
            and productivity_cur > productivity_threshold
            and income_growth_rate > 0.05
            and health_cur < health_prev
        ):
            matches.append(
                {
                    "section": "need_attention",
                    "text": "Your performance is strong, but declining health may not be sustainable long term.",
                }
            )

        # 10) Balanced workload with stable income growth
        if (
            in_eom_window
            and work_hours_week is not None
            and income_variance is not None
            and work_hours_week <= work_hour_threshold
            and income_variance < 0.10
        ):
            matches.append(
                {
                    "section": "great_job",
                    "text": "Your current workload and income pattern appears sustainable.",
                }
            )

        # 11) High income but insufficient recovery
        if (
            income_cur is not None
            and income_avg is not None
            and sleep_score is not None
            and income_cur > income_avg
            and sleep_score < sleep_threshold
        ):
            matches.append(
                {
                    "section": "need_attention",
                    "text": "Your work schedule may not allow sufficient recovery.",
                }
            )

        split_matches = self._split_cross_module_matches(matches)
        return await self._rewrite_cross_module_sections_with_llm(
            split_matches, language=language
        )

    def _split_cross_module_matches(
        self, matches: List[Dict[str, str]]
    ) -> Dict[str, List[str]]:
        """Split cross-module matches into per-section insight texts."""
        result: Dict[str, List[str]] = {"great_job": [], "need_attention": []}
        for item in matches:
            section = item.get("section")
            text = item.get("text")
            if section in result and isinstance(text, str) and text.strip():
                result[section].append(text.strip())
        return result

    def _format_cross_module_insights(self, insights: List[str]) -> str:
        """Format up to 3 cross-module insights for a section."""
        if not insights:
            return ""
        # Keep stable order while removing duplicates.
        selected = list(dict.fromkeys(insights))[:3]
        return "\n".join(f"- {item}" for item in selected)

    def _format_user_data_as_markdown(
        self, user_profile_data: Optional[Dict[str, Any]]
    ) -> str:
        if not user_profile_data:
            return ""

        markdown_sections = []

        # Basic info
        if user_profile_data.get("age"):
            markdown_sections.append(f"- Age: {user_profile_data.get('age')}")
        if user_profile_data.get("height_cm"):
            markdown_sections.append(
                f"- Height: {user_profile_data.get('height_cm')} cm"
            )
        # Targets
        targets = user_profile_data.get("targets", {})
        if targets:
            markdown_sections.append("\n**Targets:**")
            for target_field, target_info in targets.items():
                if isinstance(target_info, dict):
                    value = target_info.get("value")
                    unit = target_info.get("unit", "")
                    if value is not None:
                        markdown_sections.append(f"- {target_field}: {value} {unit}")

        # Latest health metrics
        health_data = user_profile_data.get("health_data")
        if health_data and isinstance(health_data, dict):
            markdown_sections.append("\n**Latest Health Metrics:**")
            if health_data.get("metric_date"):
                markdown_sections.append(f"- Date: {health_data.get('metric_date')}")
            if health_data.get("steps_per_day") is not None:
                markdown_sections.append(
                    f"- Steps: {health_data.get('steps_per_day')} steps"
                )
            if health_data.get("sleep_duration_hours") is not None:
                markdown_sections.append(
                    f"- Sleep: {health_data.get('sleep_duration_hours')} hours"
                )
            if health_data.get("heart_rate_avg") is not None:
                markdown_sections.append(
                    f"- Heart Rate (avg): {health_data.get('heart_rate_avg')} bpm"
                )
            if health_data.get("weight_kg") is not None:
                markdown_sections.append(f"- Weight: {health_data.get('weight_kg')} kg")
            if health_data.get("calories_burned") is not None:
                markdown_sections.append(
                    f"- Calories Burned: {health_data.get('calories_burned')} kcal"
                )

        return "\n".join(markdown_sections) if markdown_sections else ""

    def _detect_productivity_group(
        self,
        calendar_events: List[Dict[str, Any]],
        health_params: Optional[Dict[str, Any]],
        user_profile: Optional[Dict[str, Any]],
        time_data: Optional[Dict[str, Any]],
        calendar_metrics: Dict[str, Any],
    ) -> str:
        try:
            if not time_data:
                return InsightGroupConstants.CONTEXTUAL_SUGGESTION

            if self._check_safety_risks(health_params, time_data):
                return InsightGroupConstants.SAFETY_RISK
            
            if self._check_wind_down_window(time_data):
                return InsightGroupConstants.WIND_DOWN_WINDOW

            if self._check_bedtime_window(time_data):
                return InsightGroupConstants.BEDTIME_WINDOW

            if self._check_health_goals(health_params, user_profile, time_data):
                return InsightGroupConstants.HEALTH_GOAL

            if self._check_evening_flexible_window(time_data):
                return InsightGroupConstants.EVENING_FLEXIBLE

            if self._check_morning_window(time_data):
                return InsightGroupConstants.MORNING_WINDOW

            if self._check_active_window(time_data):
                return InsightGroupConstants.ACTIVE_WINDOW

            if self._check_weekend_lifestyle(time_data):
                return InsightGroupConstants.WEEKEND_LIFESTYLE

            if self._check_calendar_patterns(
                calendar_events, calendar_metrics, time_data
            ):
                return InsightGroupConstants.CALENDAR_PATTERN

            if self._check_recovery_needs(calendar_events, calendar_metrics):
                return InsightGroupConstants.RECOVERY_BREAK

            if self._check_task_type_matching(time_data):
                return InsightGroupConstants.TASK_TYPE

            return InsightGroupConstants.CONTEXTUAL_SUGGESTION

        except Exception as e:
            logger.warning(f"⚠️ Error detecting productivity group: {str(e)}")
            import traceback

            logger.debug(traceback.format_exc())
            return InsightGroupConstants.CONTEXTUAL_SUGGESTION  # Default fallback

    def _detect_overall_insight_group(
        self,
        calendar_events: List[Dict[str, Any]],
        health_params: Optional[Dict[str, Any]],
        user_profile: Optional[Dict[str, Any]],
        time_data: Optional[Dict[str, Any]],
        calendar_metrics: Dict[str, Any],
        mood: Optional[str] = None,
    ) -> str:
        try:
            if not time_data:
                return InsightGroupConstants.CONTEXTUAL_SUGGESTION

            if self._check_safety_risks(health_params, time_data):
                return InsightGroupConstants.SAFETY_RISK

            if self._check_wind_down_window(time_data):
                return InsightGroupConstants.WIND_DOWN_WINDOW

            if self._check_bedtime_window(time_data):
                return InsightGroupConstants.BEDTIME_WINDOW

            if self._check_morning_window(time_data):
                return InsightGroupConstants.MORNING_WINDOW

            if self._check_health_goals(health_params, user_profile, time_data):
                return InsightGroupConstants.HEALTH_GOAL

            if self._check_evening_flexible_window(time_data):
                return InsightGroupConstants.EVENING_FLEXIBLE

            if self._check_active_window(time_data):
                return InsightGroupConstants.ACTIVE_WINDOW

            if self._check_weekend_lifestyle(time_data):
                return InsightGroupConstants.WEEKEND_LIFESTYLE

            if self._check_recovery_needs(calendar_events, calendar_metrics):
                return InsightGroupConstants.RECOVERY_BREAK

            if self._check_task_type_matching(time_data):
                return InsightGroupConstants.TASK_TYPE

            if self._check_mood_based_rules(
                user_profile, time_data, health_params, mood
            ):
                return InsightGroupConstants.MOOD_BASED

            if self._check_evening_after_work(time_data, calendar_events):
                return InsightGroupConstants.EVENING_AFTER_WORK

            return InsightGroupConstants.CONTEXTUAL_SUGGESTION

        except Exception as e:
            logger.warning(f"⚠️ Error detecting overall insight group: {str(e)}")
            import traceback

            logger.debug(traceback.format_exc())
            return InsightGroupConstants.CONTEXTUAL_SUGGESTION  # Default fallback

    def _detect_health_insight_group(
        self,
        calendar_events: List[Dict[str, Any]],
        health_params: Optional[Dict[str, Any]],
        user_profile: Optional[Dict[str, Any]],
        time_data: Optional[Dict[str, Any]],
        calendar_metrics: Dict[str, Any],
    ) -> str:
        """Detect health insight group using productivity-like priority flow."""
        return self._detect_productivity_group(
            calendar_events=calendar_events,
            health_params=health_params,
            user_profile=user_profile,
            time_data=time_data,
            calendar_metrics=calendar_metrics,
        )

    def _check_mood_based_rules(
        self,
        user_profile: Optional[Dict[str, Any]],
        time_data: Dict[str, Any],
        health_params: Optional[Dict[str, Any]],
        mood: Optional[str] = None,
    ) -> bool:
        try:
            if not mood:
                return False

            # Check for mood-based triggers
            mood_lower = mood.lower() if mood else ""
            if mood_lower in ["terrible", "sad", "okay", "happy", "amazing"]:
                return True

            return False
        except Exception:
            return False

    def _check_weekend_lifestyle(
        self,
        time_data: Optional[Dict[str, Any]],
    ) -> bool:
        try:
            if not time_data:
                return False

            is_weekend = time_data.get("is_weekend", False)
            in_active_window = time_data.get("in_active_window", False)

            if is_weekend and in_active_window:
                # Weekend-specific checks can be added here
                return True

            return False
        except Exception:
            return False

    def _check_evening_after_work(
        self,
        time_data: Optional[Dict[str, Any]],
        calendar_events: List[Dict[str, Any]],
    ) -> bool:
        try:
            if not time_data:
                return False

            current_hour = time_data.get("current_hour", 0)
            time_to_bedtime = time_data.get("time_to_bedtime")
            in_wind_down = time_data.get("in_wind_down", False)

            # After work hours (typically 17:00-22:00)
            if 17 <= current_hour < 22:
                return True

            # Wind-down window
            if in_wind_down or (time_to_bedtime is not None and time_to_bedtime <= 60):
                return True

            return False
        except Exception:
            return False

    def _get_health_summary_by_type(
        self, health_data: Optional[Dict[str, Dict[str, Any]]], summary_type: str
    ) -> Optional[Dict[str, Any]]:
        if not health_data or not isinstance(health_data, dict):
            return None

        summary = health_data.get(summary_type)
        if summary is None:
            return None

        # If it's a Pydantic model, convert to dict
        if hasattr(summary, "model_dump"):
            return summary.model_dump()

        # Otherwise return as-is (dict)
        return summary

    def _get_latest_data_entry(
        self,
        summary: Optional[Dict[str, Any]],
        time_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not summary:
            return None
        data_array = summary.get("data", [])
        if not data_array:
            return None

        # Get current date from time_context if available
        current_date = None
        timezone = None
        if time_context:
            current_date = time_context.get(TimeDataConstants.KEY_CURRENT_DATE)
            timezone = time_context.get(TimeDataConstants.KEY_TIMEZONE)

        # Filter and sort data
        try:
            filtered_data = []
            for entry in data_array:
                entry_date_str = entry.get(APIResponseKeys.DATE, "")
                if not entry_date_str:
                    continue

                # If we have current_date, filter by matching date
                if current_date:
                    try:
                        # Parse entry date
                        entry_dt = datetime.fromisoformat(
                            entry_date_str.replace("Z", "+00:00")
                        )
                        # Convert to target timezone
                        entry_dt = entry_dt.astimezone(ZoneInfo(timezone))
                        entry_date = entry_dt.date()
                        # Only include entries matching current date
                        if entry_date == current_date:
                            filtered_data.append(entry)
                    except Exception as e:
                        logger.warning(
                            f"⚠️ Failed to parse entry date {entry_date_str}: {str(e)}"
                        )
                        # If parsing fails, include it anyway (fallback)
                        filtered_data.append(entry)
                else:
                    # No current_date, include all entries
                    filtered_data.append(entry)

            # Sort by date (descending) to get latest entry first
            if filtered_data:
                sorted_data = sorted(
                    filtered_data,
                    key=lambda x: x.get(APIResponseKeys.DATE, ""),
                    reverse=True,
                )
                return sorted_data[0] if sorted_data else None
            else:
                return None
        except Exception as e:
            logger.warning(f"⚠️ Failed to filter/sort data entries: {str(e)}")
            # Fallback to first entry if processing fails
            return data_array[0] if data_array else None

    def _calculate_calendar_metrics(
        self,
        calendar_events: List[Dict[str, Any]],
        current_dt: datetime,
        timezone: Optional[str] = None,
    ) -> Dict[str, Any]:
        back_to_back_count = 0
        meeting_minutes = 0

        if not calendar_events:
            return {"back_to_back_count": 0, "meeting_minutes": 0}

        tz = ZoneInfo(timezone) if timezone else ZoneInfo("UTC")

        # Calculate back-to-back count
        for i in range(len(calendar_events) - 1):
            event1 = calendar_events[i]
            event2 = calendar_events[i + 1]
            end_time1 = event1.get("endTime")
            start_time2 = event2.get("startTime")

            if end_time1 and start_time2:
                try:
                    end1 = datetime.fromisoformat(end_time1.replace("Z", "+00:00"))
                    start2 = datetime.fromisoformat(start_time2.replace("Z", "+00:00"))
                    if end1.tzinfo is None:
                        end1 = end1.replace(tzinfo=tz)
                    if start2.tzinfo is None:
                        start2 = start2.replace(tzinfo=tz)

                    # Check if events are back-to-back (no gap or very small gap < 5 mins)
                    gap_minutes = (start2 - end1).total_seconds() / 60
                    if 0 <= gap_minutes < 5:
                        back_to_back_count += 1
                except Exception:
                    # Fallback to string comparison
                    if end_time1 == start_time2:
                        back_to_back_count += 1

        # Calculate meeting minutes in last 3 hours
        try:
            three_hours_ago = current_dt - timedelta(hours=3)
            for event in calendar_events:
                start_time_str = event.get("startTime")
                end_time_str = event.get("endTime")
                if start_time_str and end_time_str:
                    try:
                        event_start = datetime.fromisoformat(
                            start_time_str.replace("Z", "+00:00")
                        )
                        event_end = datetime.fromisoformat(
                            end_time_str.replace("Z", "+00:00")
                        )
                        if event_start.tzinfo is None:
                            event_start = event_start.replace(tzinfo=tz)
                        if event_end.tzinfo is None:
                            event_end = event_end.replace(tzinfo=tz)

                        # Check if event overlaps with last 3 hours
                        if event_end > three_hours_ago and event_start < current_dt:
                            overlap_start = max(event_start, three_hours_ago)
                            overlap_end = min(event_end, current_dt)
                            meeting_minutes += (
                                overlap_end - overlap_start
                            ).total_seconds() / 60
                    except Exception:
                        continue
        except Exception:
            pass

        return {
            "back_to_back_count": back_to_back_count,
            "meeting_minutes": meeting_minutes,
        }

    def _check_safety_risks(
        self,
        health_params: Optional[Dict[str, Any]],
        time_data: Dict[str, Any],
    ) -> bool:
        if not health_params:
            return False
        logger.info(f"time_data: {time_data}")
        current_hour = time_data[TimeDataConstants.KEY_CURRENT_HOUR]
        sleep_target = time_data.get(TimeDataConstants.KEY_SLEEP_TARGET, {})

        # Get values from health_params
        latest_heart_rate = health_params.get(HealthDataConstants.KEY_LATEST_HEART_RATE)
        resting_heart_rate = health_params.get(
            HealthDataConstants.KEY_RESTING_HEART_RATE
        )
        sleep_lastnight = health_params.get(HealthDataConstants.KEY_SLEEP_LASTNIGHT)
        sleep_goal = health_params.get(HealthDataConstants.KEY_SLEEP_GOAL)

        # CRITICAL safety risk: Dangerous heart rate
        if latest_heart_rate and (latest_heart_rate < 40 or latest_heart_rate > 180):
            logger.info("🚨 SAFETY RISK: Dangerous heart rate detected")
            return True

        # CRITICAL safety risk: Dangerously high resting HR (potential health issue)
        if resting_heart_rate and resting_heart_rate > 100:
            logger.info("🚨 SAFETY RISK: Dangerously high resting heart rate")
            return True

        return False

    def _check_wind_down_window(self, time_data: Dict[str, Any]) -> bool:
        time_to_bedtime = time_data.get(TimeDataConstants.KEY_TIME_TO_BEDTIME)

        # Wind-down: time_to_bedtime <= 60 (1 hour before bedtime)
        if time_to_bedtime is not None and time_to_bedtime <= 60:
            return True

        # Late wind-down: time_to_bedtime <= 30
        if time_to_bedtime is not None and time_to_bedtime <= 30:
            return True

        return False

    def _check_bedtime_window(self, time_data: Dict[str, Any]) -> bool:
        bedtime_start_str = time_data.get(TimeDataConstants.KEY_BEDTIME_START_STR)
        bedtime_end_str = time_data.get(TimeDataConstants.KEY_BEDTIME_END_STR)
        current_hour = time_data.get(TimeDataConstants.KEY_CURRENT_HOUR)
        current_minute = time_data.get(TimeDataConstants.KEY_CURRENT_MINUTE)

        # Bedtime window: isInWindow(now, bedtime_start, bedtime_end)
        if bedtime_start_str and bedtime_end_str:
            try:
                bedtime_start_parts = bedtime_start_str.split(":")
                bedtime_end_parts = bedtime_end_str.split(":")
                bedtime_start_hour = int(bedtime_start_parts[0])
                bedtime_start_min = (
                    int(bedtime_start_parts[1]) if len(bedtime_start_parts) > 1 else 0
                )
                bedtime_end_hour = int(bedtime_end_parts[0])
                bedtime_end_min = (
                    int(bedtime_end_parts[1]) if len(bedtime_end_parts) > 1 else 0
                )

                current_time_minutes = current_hour * 60 + current_minute
                bedtime_start_minutes = bedtime_start_hour * 60 + bedtime_start_min
                bedtime_end_minutes = bedtime_end_hour * 60 + bedtime_end_min

                # Handle wrap-around (e.g., 22:00 to 08:00)
                if bedtime_start_minutes > bedtime_end_minutes:
                    in_bedtime_window = (
                        current_time_minutes >= bedtime_start_minutes
                    ) or (current_time_minutes <= bedtime_end_minutes)
                else:
                    in_bedtime_window = (
                        bedtime_start_minutes
                        <= current_time_minutes
                        <= bedtime_end_minutes
                    )

                if in_bedtime_window:
                    return True
            except Exception:
                pass

        return False

    def _check_bedtime_windows(self, time_data: Dict[str, Any]) -> bool:
        # Check wind-down window
        if self._check_wind_down_window(time_data):
            return True

        # Check bedtime window
        if self._check_bedtime_window(time_data):
            return True

        return False

    def _check_morning_window(self, time_data: Dict[str, Any]) -> bool:
        bedtime_end_str = time_data.get(TimeDataConstants.KEY_BEDTIME_END_STR)
        active_start_str = time_data.get(TimeDataConstants.KEY_ACTIVE_START_STR)
        current_hour = time_data.get(TimeDataConstants.KEY_CURRENT_HOUR)
        current_minute = time_data.get(TimeDataConstants.KEY_CURRENT_MINUTE)

        if not bedtime_end_str or not active_start_str:
            return False

        try:
            # Parse bedtime_end
            bedtime_end_parts = bedtime_end_str.split(":")
            bedtime_end_hour = int(bedtime_end_parts[0])
            bedtime_end_min = (
                int(bedtime_end_parts[1]) if len(bedtime_end_parts) > 1 else 0
            )

            # Parse active_hours_start
            active_start_parts = active_start_str.split(":")
            active_start_hour = int(active_start_parts[0])
            active_start_min = (
                int(active_start_parts[1]) if len(active_start_parts) > 1 else 0
            )

            current_time_minutes = current_hour * 60 + current_minute
            bedtime_end_minutes = bedtime_end_hour * 60 + bedtime_end_min
            active_start_minutes = active_start_hour * 60 + active_start_min

            # Check if current_time is after bedtime_end and before active_hours_start
            # Handle wrap-around case (e.g., bedtime_end = 08:00, active_start = 09:00)
            if bedtime_end_minutes <= active_start_minutes:
                # Normal case: bedtime_end < active_start (e.g., 08:00 < 09:00)
                in_morning_window = (
                    bedtime_end_minutes < current_time_minutes < active_start_minutes
                )
            else:
                # Wrap-around case: bedtime_end > active_start (e.g., 22:00 > 07:00)
                # Morning window is from bedtime_end to midnight OR from midnight to active_start
                in_morning_window = (current_time_minutes >= bedtime_end_minutes) or (
                    current_time_minutes < active_start_minutes
                )

            if in_morning_window:
                return True
        except Exception as e:
            logger.warning(f"Error checking morning window: {e}")
            pass

        return False

    def _check_evening_flexible_window(self, time_data: Dict[str, Any]) -> bool:
        bedtime_start_str = time_data.get(TimeDataConstants.KEY_BEDTIME_START_STR)
        current_hour = time_data.get(TimeDataConstants.KEY_CURRENT_HOUR)
        current_minute = time_data.get(TimeDataConstants.KEY_CURRENT_MINUTE)

        if not bedtime_start_str:
            return False

        try:
            # Parse bedtime_start
            bedtime_start_parts = bedtime_start_str.split(":")
            bedtime_start_hour = int(bedtime_start_parts[0])
            bedtime_start_min = (
                int(bedtime_start_parts[1]) if len(bedtime_start_parts) > 1 else 0
            )

            # Calculate wind_down_start = bedtime_start - 1 hours
            bedtime_start_minutes = bedtime_start_hour * 60 + bedtime_start_min
            wind_down_start_minutes = bedtime_start_minutes - 60  # 1 hour = 60 minutes

            # Handle wrap-around if wind_down_start goes negative
            if wind_down_start_minutes < 0:
                wind_down_start_minutes += 24 * 60  # Add 24 hours

            # Evening flexible window: from 19:00 (1140 minutes) to wind_down_start
            evening_start_minutes = 19 * 60  # 19:00 = 1140 minutes
            current_time_minutes = current_hour * 60 + current_minute

            # Check if current_time is >= 19:00 and < wind_down_start
            # Handle wrap-around case (e.g., wind_down_start might be next day)
            in_evening_flexible = False
            if wind_down_start_minutes > evening_start_minutes:
                # Normal case: 19:00 < wind_down_start (e.g., 19:00 < 20:00)
                in_evening_flexible = (
                    evening_start_minutes
                    <= current_time_minutes
                    < wind_down_start_minutes
                )

            if in_evening_flexible:
                return True
            elif wind_down_start_minutes <= evening_start_minutes:
                logger.debug(
                    f"Evening flexible window skipped: wind_down_start ({wind_down_start_minutes}m) "
                    f"<= evening_start (1140m). bedtime_start={bedtime_start_str}"
                )
        except Exception as e:
            logger.warning(f"Error checking evening flexible window: {e}")
            pass

        return False

    def _check_active_window(self, time_data: Dict[str, Any]) -> bool:
        time_to_next_event = time_data.get(TimeDataConstants.KEY_TIME_TO_NEXT_EVENT)
        in_active_window = time_data.get(TimeDataConstants.KEY_IN_ACTIVE_WINDOW, False)
        is_weekend = time_data.get(TimeDataConstants.KEY_IS_WEEKEND, False)
        logger.info(f"time_metrics: {time_data}")

        # Prep window: timeToNextEvent <= 30
        if time_to_next_event is not None and time_to_next_event <= 30:
            return True

        # Active hours: inActiveWindow
        if in_active_window and not is_weekend:
            return True

        return False

    def _check_calendar_patterns(
        self,
        calendar_events: List[Dict[str, Any]],
        calendar_metrics: Dict[str, Any],
        time_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if not calendar_events:
            return False

        back_to_back_count = calendar_metrics.get(
            CalendarDataConstants.KEY_BACK_TO_BACK_COUNT, 0
        )
        meeting_minutes = calendar_metrics.get(
            CalendarDataConstants.KEY_MEETING_MINUTES, 0
        )
        calendar_density = calendar_metrics.get(
            CalendarDataConstants.KEY_CALENDAR_DENSITY, 0
        )

        # Back-to-back: backToBack >= 2
        if back_to_back_count >= 2:
            return True

        # Meeting streak: meetingMinutesLast3Hours >= 120
        if meeting_minutes >= 120:
            return True

        # Calendar density: High number of meetings in active hours
        # Consider high density if >= 5 meetings in active hours
        if calendar_density >= 5:
            return True

        # Note: "no focus blocks in active window for >=3 days" requires historical data
        # This would need to be tracked over time, not just current day
        # For now, we skip this check as it requires additional data storage

        return False

    def _check_health_goals(
        self,
        health_params: Optional[Dict[str, Any]],
        user_profile: Optional[Dict[str, Any]],
        time_data: Dict[str, Any],
    ) -> bool:
        if not health_params or not user_profile:
            return False

        current_hour = time_data[TimeDataConstants.KEY_CURRENT_HOUR]
        is_weekend = time_data[TimeDataConstants.KEY_IS_WEEKEND]
        free_slot_length = time_data.get(TimeDataConstants.KEY_FREE_SLOT_LENGTH)

        # Get values from health_params
        steps_today = health_params.get(HealthDataConstants.KEY_STEPS_TODAY)
        steps_goal = health_params.get(HealthDataConstants.KEY_STEPS_GOAL)
        daily_health_progress = health_params.get(
            HealthDataConstants.KEY_DAILY_HEALTH_PROGRESS
        )
        weekly_health_progress = health_params.get(
            HealthDataConstants.KEY_WEEKLY_HEALTH_PROGRESS
        )

        has_sufficient_slot = free_slot_length is None or free_slot_length >= 30

        # Daily health catch-up: dailyHealthProgress < target AND (5 PM - 7 PM OR weekend) AND slot >= 30
        if (
            daily_health_progress is not None
            and daily_health_progress.get(
                HealthDataConstants.KEY_PROGRESS_PERCENTAGE, 100.0
            )
            < 100.0
        ):
            in_catchup_window = 17 <= current_hour < 19  # 5 PM - 7 PM
            if (in_catchup_window or is_weekend) and has_sufficient_slot:
                return True

        # Weekly health catch-up: weeklyHealthProgress < target AND weekend AND slot >= 30
        if (
            is_weekend
            and weekly_health_progress is not None
            and weekly_health_progress.get(
                HealthDataConstants.KEY_PROGRESS_PERCENTAGE, 100.0
            )
            < 100.0
            and has_sufficient_slot
        ):
            return True

        # Fallback: Check steps_today vs steps_goal (for backward compatibility)
        if steps_today and steps_goal and steps_today < steps_goal:
            in_catchup_window = 17 <= current_hour < 19  # 5 PM - 7 PM
            if (in_catchup_window or is_weekend) and has_sufficient_slot:
                return True

        return False

    def _check_recovery_needs(
        self, calendar_events: List[Dict[str, Any]], calendar_metrics: Dict[str, Any]
    ) -> bool:
        if not calendar_events:
            return False

        back_to_back_count = calendar_metrics.get(
            CalendarDataConstants.KEY_BACK_TO_BACK_COUNT, 0
        )
        meeting_minutes = calendar_metrics.get(
            CalendarDataConstants.KEY_MEETING_MINUTES, 0
        )
        continuous_events_minutes = calendar_metrics.get(
            CalendarDataConstants.KEY_CONTINUOUS_EVENTS_MINUTES, 0
        )

        # Back-to-back: backToBack >= 2 (micro recovery)
        if back_to_back_count >= 2:
            return True

        # Meeting streak: meetingMinutesLast3Hours >= 120 (recovery window)
        if meeting_minutes >= 120:
            return True

        # Continuous events: continuousEvents >= 90 mins (break nudge)
        # Note: continuous_events_minutes already checks for >= 90 mins threshold
        if continuous_events_minutes >= 90:
            return True

        # Recovery missing: continuousEvents >= 120
        if continuous_events_minutes >= 120:
            return True

        # Note: breaks_today < 2 check requires tracking breaks over time
        # This would need additional data storage, skipping for now

        return False

    def _check_task_type_matching(self, time_data: Dict[str, Any]) -> bool:
        free_slot_length = time_data.get("free_slot_length")
        time_to_next_event = time_data.get("time_to_next_event")
        in_active_window = time_data.get("in_active_window", False)

        if free_slot_length is None:
            return False

        # Deep work requires >= 45 mins AND in active window AND >= 20 mins before next event
        if free_slot_length < 45 and in_active_window:
            return True

        # Prep window: timeToNextEvent <= 20
        if time_to_next_event is not None and time_to_next_event <= 20:
            return True

        return False

    def _extract_user_profile_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Extract goals and process them
        goals = data.get("goals", [])

        # Build profile data
        user_data = {
            "age": data.get("age"),
            "height_cm": data.get("height_cm"),
            "bedtime_start": data.get("bedtime_start"),
            "bedtime_end": data.get("bedtime_end"),
            "active_hours_start_time": data.get("active_hours_start_time"),
            "active_hours_end_time": data.get("active_hours_end_time"),
        }
        targets_data = self._extract_user_targets(goals)
        if targets_data:
            user_data["targets"] = targets_data

        return user_data

    async def _build_day_context(
        self,
        user_id: str,
        health_params: Optional[Dict[str, Any]],
        time_data: Optional[Dict[str, Any]],
        calendar_metrics: Optional[Dict[str, Any]],
        calendar_events: Optional[List[Dict[str, Any]]],
        language: str = "en",
    ) -> str:
        try:
            current_date = (time_data or {}).get("current_date", "")
            cache_key = f"day_context:{user_id}:{current_date}"
            cached = await self.redis_client.get_data(cache_key)
            if cached:
                return cached.get("context", "")

            system_prompt, user_prompt = get_day_context_prompt(
                health_params=health_params,
                time_data=time_data,
                calendar_metrics=calendar_metrics,
                calendar_events=calendar_events,
                language=language,
            )
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
            response = await self.llm.ainvoke(messages)
            context_str = llm_response_text(response)
            await self.redis_client.set_data(
                cache_key, {"context": context_str}, expire=300
            )
            logger.info(f"📅 Built day context for user {user_id}")
            return context_str
        except Exception as e:
            logger.warning(f"⚠️ Failed to build day context for user {user_id}: {e}")
            return ""

    async def analyze_productivity(
        self,
        user_id: str,
        timezone: Optional[str] = None,
        provider_name: Optional[str] = "",
        language: Optional[str] = "en-US",
        force_update: bool = False,
    ) -> Dict[str, Any]:
        try:
            # Parse current time
            if timezone:
                tz = ZoneInfo(timezone)
                current_dt = datetime.now(tz)
            else:
                tz = ZoneInfo("UTC")
                current_dt = datetime.now(tz)

            user_language = language or "en-US"
            gen_language = resolve_generation_language(user_language)

            logger.info(
                f"🔍 Starting productivity analysis for user {user_id} (timezone: {timezone}, provider: {provider_name})"
            )

            # Step 1: Get user profile data first (needed for calendar metrics)
            user_profile_data = None
            try:
                general_data = await CacheHelpers.fetch_general_user_data_cached(
                    self.redis_client,
                    self.external_api_service,
                    user_id,
                    log_prefix="Productivity insight",
                )
                if general_data:
                    user_profile_data = self._extract_user_profile_data(general_data)
            except Exception as e:
                logger.warning(f"⚠️ Failed to fetch user profile: {str(e)}")

            # Step 2: Prepare calendar data (with user_profile for active hours)
            prepare_calendar = PrepareCalendarData(
                external_api_service=self.external_api_service,
                user_id=user_id,
                provider_name=provider_name,
                timezone=timezone,
                current_dt=current_dt,
                user_profile=user_profile_data,
            )
            calendar_data = await prepare_calendar.prepare_all_calendar_data()

            events = calendar_data.get(CalendarDataConstants.KEY_EVENTS, [])
            formatted_calendar_data = calendar_data.get(
                CalendarDataConstants.KEY_FORMATTED_CALENDAR_DATA, ""
            )
            calendar_metrics = calendar_data.get(
                CalendarDataConstants.KEY_CALENDAR_METRICS, {}
            )

            # Step 3: Get health data from /api/health/summaries endpoint (current week).
            logger.info(f"📊 Fetching health summaries for user {user_id}")
            health_data = None
            try:
                if self.external_api_service:
                    health_data = await self.external_api_service.get_health_summaries(
                        user_id=user_id,
                        timezone=timezone,
                    )
                    if health_data:
                        logger.info(f"✅ Successfully retrieved health summaries")
                    else:
                        logger.warning(f"⚠️ No health data available for user {user_id}")
                else:
                    logger.warning(
                        "⚠️ External API service not available for health data"
                    )
            except Exception as e:
                logger.warning(f"⚠️ Failed to fetch health data: {str(e)}")

            # Step 4: Prepare all time data using PrepareTimeData
            prepare_time = PrepareTimeData(
                user_profile=user_profile_data,
                current_dt=current_dt,
                tz=tz,
                timezone=timezone,
                calendar_events=events or [],
            )
            time_data = (
                prepare_time.prepare_all_time_data()
            )  # Gộp time_context và time_metrics

            # Step 5: Prepare health data and detect which productivity group applies
            health_params = {}
            try:
                if self.external_api_service:
                    prepare_health = PrepareHealthData(
                        external_api_service=self.external_api_service,
                        user_id=user_id,
                        user_profile=user_profile_data,
                        timezone=timezone,
                        time_data=time_data,
                    )
                    health_params = await prepare_health.prepare_all_health_params()
                    logger.info(f"✅ Successfully prepared all health parameters")
            except Exception as e:
                logger.warning(f"⚠️ Failed to prepare health data: {str(e)}")
            group = self._detect_productivity_group(
                calendar_events=events or [],
                health_params=health_params,
                user_profile=user_profile_data,
                time_data=time_data,  # Gộp time_context và time_metrics
                calendar_metrics=calendar_metrics,
            )
            logger.info(f"🎯 Detected productivity group: {group}")

            # Step 6: Get appropriate prompt based on group
            # Extract user profile targets for prompts
            targets = user_profile_data.get("targets", {}) if user_profile_data else {}
            bedtime_start = user_profile_data.get("bedtime_start")
            bedtime_end = user_profile_data.get("bedtime_end")
            active_start = user_profile_data.get("active_hours_start_time")
            active_end = user_profile_data.get("active_hours_end_time")
            break_interval = (
                targets.get("break_interval_hours", {}).get("value")
                if isinstance(targets.get("break_interval_hours", {}), dict)
                else None
            )
            break_duration = (
                targets.get("break_duration_min", {}).get("value")
                if isinstance(targets.get("break_duration_min", {}), dict)
                else None
            )

            # Extract health data values for prompts from health_params
            steps_today = health_params.get(HealthDataConstants.KEY_STEPS_TODAY)
            steps_goal = health_params.get(HealthDataConstants.KEY_STEPS_GOAL)
            sleep_lastnight = health_params.get(HealthDataConstants.KEY_SLEEP_LASTNIGHT)
            sleep_goal = health_params.get(HealthDataConstants.KEY_SLEEP_GOAL)
            latest_heart_rate = health_params.get(
                HealthDataConstants.KEY_LATEST_HEART_RATE
            )
            resting_heart_rate = health_params.get(
                HealthDataConstants.KEY_RESTING_HEART_RATE
            )
            baseline_resting_hr = health_params.get(
                HealthDataConstants.KEY_BASELINE_RESTING_HR
            )
            sleep_last3nights = health_params.get(
                HealthDataConstants.KEY_SLEEP_LAST3NIGHTS
            )
            daily_health_progress = health_params.get(
                HealthDataConstants.KEY_DAILY_HEALTH_PROGRESS
            )
            weekly_health_progress = health_params.get(
                HealthDataConstants.KEY_WEEKLY_HEALTH_PROGRESS
            )

            # Extract calculated values from time_data (gộp time_context và time_metrics)
            time_to_next_event = time_data.get(TimeDataConstants.KEY_TIME_TO_NEXT_EVENT)
            free_slot_length = time_data.get(TimeDataConstants.KEY_FREE_SLOT_LENGTH)
            slot_length = free_slot_length  # Same as free_slot_length
            time_to_bedtime = time_data.get(TimeDataConstants.KEY_TIME_TO_BEDTIME)
            in_active_window = time_data.get(TimeDataConstants.KEY_IN_ACTIVE_WINDOW)
            is_weekend = time_data.get(TimeDataConstants.KEY_IS_WEEKEND, False)
            current_time = (
                time_data.get(TimeDataConstants.KEY_CURRENT_TIME_ISO)
                or current_dt.isoformat()
            )

            # Extract calendar metrics
            back_to_back_count = calendar_metrics.get(
                CalendarDataConstants.KEY_BACK_TO_BACK_COUNT
            )
            meeting_minutes = calendar_metrics.get(
                CalendarDataConstants.KEY_MEETING_MINUTES
            )
            calendar_density = calendar_metrics.get(
                CalendarDataConstants.KEY_CALENDAR_DENSITY
            )
            continuous_events_minutes = calendar_metrics.get(
                CalendarDataConstants.KEY_CONTINUOUS_EVENTS_MINUTES
            )
            back_to_back_count = calendar_metrics.get(
                CalendarDataConstants.KEY_BACK_TO_BACK_COUNT
            )
            meeting_minutes = calendar_metrics.get(
                CalendarDataConstants.KEY_MEETING_MINUTES
            )
            calendar_density = calendar_metrics.get(
                CalendarDataConstants.KEY_CALENDAR_DENSITY
            )
            continuous_events_minutes = calendar_metrics.get(
                CalendarDataConstants.KEY_CONTINUOUS_EVENTS_MINUTES
            )
            overall_key = self._get_cache_key(
                user_id, user_language, insight_type="overall"
            )
            overall_insight = await self.redis_client.get_data(overall_key)

            overall_context = ""
            if overall_insight:
                overall_context = overall_insight.get("insight")

            # Build shared day context for cross-insight coherence
            day_context = await self._build_day_context(
                user_id=user_id,
                health_params=health_params,
                time_data=time_data,
                calendar_metrics=calendar_metrics,
                calendar_events=events or [],
                language=gen_language,
            )

            # Today-only reminders: merged into calendar_data for groups whose
            # prompts include a Calendar Events section (not appended at the end).
            reminders_today = await self._fetch_today_reminders(
                user_id=user_id,
                timezone=timezone,
                current_dt=current_dt,
            )
            reminders_section = self._build_reminders_section(reminders_today, timezone)
            calendar_for_prompt = self._merge_calendar_with_reminders(
                formatted_calendar_data, reminders_section
            )

            if group == InsightGroupConstants.SAFETY_RISK:
                system_prompt, user_prompt = get_safety_risk_insight_prompt(
                    latest_heart_rate=latest_heart_rate,
                    resting_heart_rate=resting_heart_rate,
                    sleep_lastnight=sleep_lastnight,
                    sleep_goal=sleep_goal,
                    current_time=current_time,
                    timezone=timezone,
                    baseline_resting_hr=baseline_resting_hr,
                    sleep_last3nights=sleep_last3nights,
                    language=gen_language,
                    overall_context=overall_context,
                )
            elif group == InsightGroupConstants.WIND_DOWN_WINDOW:
                sleep_quality = health_params.get(HealthDataConstants.KEY_SLEEP_QUALITY)
                wind_down_buffer_mins = health_params.get(
                    HealthDataConstants.KEY_WIND_DOWN_BUFFER_MINS
                )
                system_prompt, user_prompt = get_wind_down_window_insight_prompt(
                    calendar_data=calendar_for_prompt,
                    current_time=current_time,
                    bedtime_start=bedtime_start,
                    bedtime_end=bedtime_end,
                    time_to_bedtime=time_to_bedtime,
                    sleep_lastnight=sleep_lastnight,
                    sleep_quality=sleep_quality,
                    sleep_goal=sleep_goal,
                    wind_down_buffer_mins=wind_down_buffer_mins,
                    timezone=timezone,
                    language=gen_language,
                    overall_context=overall_context,
                )
            elif group == InsightGroupConstants.BEDTIME_WINDOW:
                sleep_quality = health_params.get(HealthDataConstants.KEY_SLEEP_QUALITY)
                bedtime_streak = health_params.get(
                    HealthDataConstants.KEY_BEDTIME_STREAK
                )
                last_wake_time = health_params.get(
                    HealthDataConstants.KEY_LAST_WAKE_TIME
                )
                first_sleep_time = health_params.get(
                    HealthDataConstants.KEY_FIRST_SLEEP_TIME
                )
                all_3_nights_have_data = health_params.get(
                    HealthDataConstants.KEY_ALL_3_NIGHTS_HAVE_DATA, False
                )
                # Reconstruct sleep_last3nights dict for formatting
                sleep_last3nights_dict = None
                if sleep_last3nights is not None:
                    sleep_last3nights_dict = {
                        "sleep_hours": sleep_last3nights,
                        "all_3_nights_have_data": all_3_nights_have_data,
                    }
                system_prompt, user_prompt = get_bedtime_window_insight_prompt(
                    calendar_data=calendar_for_prompt,
                    current_time=current_time,
                    bedtime_start=bedtime_start,
                    bedtime_end=bedtime_end,
                    sleep_lastnight=sleep_lastnight,
                    sleep_quality=sleep_quality,
                    sleep_last3nights=sleep_last3nights,
                    bedtime_streak=bedtime_streak,
                    last_wake_time=last_wake_time,
                    first_sleep_time=first_sleep_time,
                    sleep_goal=sleep_goal,
                    timezone=timezone,
                    language=gen_language,
                    overall_context=overall_context,
                )
            elif group == InsightGroupConstants.MORNING_WINDOW:
                sleep_quality = health_params.get(HealthDataConstants.KEY_SLEEP_QUALITY)
                system_prompt, user_prompt = get_morning_start_insight_prompt(
                    calendar_data=calendar_for_prompt,
                    current_time=current_time,
                    timezone=timezone,
                    bedtime_end=bedtime_end,
                    active_hours_start=active_start,
                    sleep_lastnight=sleep_lastnight,
                    sleep_quality=sleep_quality,
                    sleep_goal=sleep_goal,
                    time_to_next_event=time_to_next_event,
                    free_slot_length=free_slot_length,
                    language=gen_language,
                    overall_context=overall_context,
                )
            elif group == InsightGroupConstants.ACTIVE_WINDOW:
                system_prompt, user_prompt = get_active_window_insight_prompt(
                    calendar_data=calendar_for_prompt,
                    current_time=current_time,
                    active_hours_start=active_start,
                    active_hours_end=active_end,
                    time_to_next_event=time_to_next_event,
                    free_slot_length=free_slot_length,
                    calendar_density=calendar_density,
                    back_to_back_count=back_to_back_count,
                    timezone=timezone,
                    language=gen_language,
                    overall_context=overall_context,
                )
            elif group == InsightGroupConstants.CALENDAR_PATTERN:
                system_prompt, user_prompt = get_calendar_pattern_insight_prompt(
                    calendar_data=calendar_for_prompt,
                    current_time=current_time,
                    timezone=timezone,
                    active_hours_start=active_start,
                    active_hours_end=active_end,
                    calendar_density=calendar_density,
                    back_to_back_count=back_to_back_count,
                    meeting_minutes=meeting_minutes,
                    language=gen_language,
                    overall_context=overall_context,
                )
            elif group == InsightGroupConstants.HEALTH_GOAL:
                system_prompt, user_prompt = get_health_goal_insight_prompt(
                    steps_today=steps_today,
                    steps_goal=steps_goal,
                    sleep_lastnight=sleep_lastnight,
                    sleep_goal=sleep_goal,
                    latest_heart_rate=latest_heart_rate,
                    resting_heart_rate=resting_heart_rate,
                    current_time=current_time,
                    timezone=timezone,
                    free_slot_length=free_slot_length,
                    daily_health_progress=daily_health_progress,
                    weekly_health_progress=weekly_health_progress,
                    is_weekend=is_weekend,
                    language=gen_language,
                    overall_context=overall_context,
                )
            elif group == InsightGroupConstants.EVENING_FLEXIBLE:
                sleep_quality = health_params.get(HealthDataConstants.KEY_SLEEP_QUALITY)
                wind_down_buffer_mins = health_params.get(
                    HealthDataConstants.KEY_WIND_DOWN_BUFFER_MINS
                )
                system_prompt, user_prompt = get_evening_flexible_insight_prompt(
                    calendar_data=calendar_for_prompt,
                    current_time=current_time,
                    timezone=timezone,
                    bedtime_start=bedtime_start,
                    time_to_bedtime=time_to_bedtime,
                    wind_down_buffer_mins=wind_down_buffer_mins,
                    sleep_lastnight=sleep_lastnight,
                    sleep_quality=sleep_quality,
                    free_slot_length=free_slot_length,
                    language=gen_language,
                    overall_context=overall_context,
                )
            elif group == InsightGroupConstants.RECOVERY_BREAK:
                system_prompt, user_prompt = get_recovery_break_insight_prompt(
                    calendar_data=calendar_for_prompt,
                    current_time=current_time,
                    timezone=timezone,
                    break_interval=break_interval,
                    break_duration=break_duration,
                    back_to_back_count=back_to_back_count,
                    meeting_minutes=meeting_minutes,
                    continuous_events_minutes=continuous_events_minutes,
                    language=gen_language,
                    overall_context=overall_context,
                )
            elif group == InsightGroupConstants.TASK_TYPE:
                system_prompt, user_prompt = get_task_type_insight_prompt(
                    calendar_data=calendar_for_prompt,
                    current_time=current_time,
                    timezone=timezone,
                    slot_length=slot_length,
                    time_to_next_event=time_to_next_event,
                    time_to_bedtime=time_to_bedtime,
                    in_active_window=in_active_window,
                    language=gen_language,
                    overall_context=overall_context,
                )
            elif group == InsightGroupConstants.WEEKEND_LIFESTYLE:
                system_prompt, user_prompt = get_weekend_lifestyle_insight_prompt(
                    calendar_data=calendar_for_prompt,
                    current_time=current_time,
                    timezone=timezone,
                    language=gen_language,
                    free_slot_length=free_slot_length,
                    steps_today=steps_today,
                    steps_goal=steps_goal,
                    overall_context=overall_context,
                )
            else:  # CONTEXTUAL_SUGGESTION (default)
                # Format time_context and calendar_patterns for prompt
                time_context_for_prompt = {
                    TimeDataConstants.KEY_CURRENT_HOUR: time_data.get(
                        TimeDataConstants.KEY_CURRENT_HOUR
                    ),
                    TimeDataConstants.KEY_CURRENT_MINUTE: time_data.get(
                        TimeDataConstants.KEY_CURRENT_MINUTE
                    ),
                    TimeDataConstants.KEY_IS_WEEKEND: time_data.get(
                        TimeDataConstants.KEY_IS_WEEKEND, False
                    ),
                    TimeDataConstants.KEY_TIME_TO_BEDTIME: time_to_bedtime,
                    TimeDataConstants.KEY_TIME_TO_NEXT_EVENT: time_to_next_event,
                    TimeDataConstants.KEY_IN_ACTIVE_WINDOW: in_active_window,
                    TimeDataConstants.KEY_FREE_SLOT_LENGTH: free_slot_length,
                }
                calendar_patterns_for_prompt = {
                    CalendarDataConstants.KEY_BACK_TO_BACK_COUNT: calendar_metrics.get(
                        CalendarDataConstants.KEY_BACK_TO_BACK_COUNT
                    ),
                    CalendarDataConstants.KEY_MEETING_MINUTES: calendar_metrics.get(
                        CalendarDataConstants.KEY_MEETING_MINUTES
                    ),
                }
                system_prompt, user_prompt = get_contextual_suggestion_insight_prompt(
                    calendar_data=calendar_for_prompt,
                    health_params=health_params,
                    user_profile=user_profile_data,
                    current_time=current_time,
                    timezone=timezone,
                    time_context=time_context_for_prompt,
                    calendar_patterns=calendar_patterns_for_prompt,
                    language=gen_language,
                    overall_context=overall_context,
                )

            # Step 4: Call LLM
            if day_context:
                system_prompt = (
                    system_prompt
                    + "\n\n**Day Profile (shared context):**\n"
                    + day_context
                )
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]

            response = await self.llm.ainvoke(messages)

            # Extract text from response and parse JSON using helper methods
            content_text = llm_response_text(response)
            parsed_insight = self._parse_json_response(content_text)
            parsed_insight = await translate_insight_dict(
                parsed_insight,
                user_language,
                self.llm,
                insight_type="productivity_insight",
            )

            # Force Vietnamese wording consistency for wind-down terms
            for key, value in parsed_insight.items():
                if isinstance(value, str):
                    parsed_insight[key] = self._normalize_wind_down_terms(
                        value, user_language
                    )

            # Convert parsed dict to JSON string for storage
            insight_text = json.dumps(parsed_insight, ensure_ascii=False)

            logger.info(
                f"✅ Successfully generated productivity insight for user {user_id}"
            )

            return {
                "status": "success",
                "user_id": user_id,
                "insight": insight_text,
                "error": None,
            }

        except Exception as e:
            logger.error(
                f"❌ Error analyzing productivity for user {user_id}: {str(e)}"
            )
            import traceback

            logger.error(traceback.format_exc())
            return {
                "status": "error",
                "user_id": user_id,
                "insight": None,
                "error": f"Error during productivity analysis: {str(e)}",
            }

    async def analyze_health_insight(
        self,
        user_id: str,
        timezone: Optional[str] = None,
        provider_name: Optional[str] = "",
        language: Optional[str] = "en-US",
        force_update: Optional[bool] = False,
    ) -> Dict[str, Any]:
        """
        Analyze user health insight in exactly two points:
        - point1: health summary
        - point2: gentle suggestion
        """
        try:
            tz = ZoneInfo(timezone) if timezone else ZoneInfo("UTC")
            current_dt = datetime.now(tz)

            user_language = language or "en-US"
            gen_language = resolve_generation_language(user_language)
            cache_key = self._get_cache_key(
                user_id, user_language, insight_type="health_insight"
            )

            if not force_update:
                cached_result = await self.redis_client.get_data(cache_key)
                if cached_result:
                    return {
                        "status": "success",
                        "user_id": user_id,
                        "insight": cached_result.get("insight"),
                        "error": None,
                    }

            # Step 1: Load user profile (needed for health goals inside PrepareHealthData)
            user_profile_data = None
            try:
                general_data = await CacheHelpers.fetch_general_user_data_cached(
                    self.redis_client,
                    self.external_api_service,
                    user_id,
                    log_prefix="Health insight",
                )
                if general_data:
                    user_profile_data = self._extract_user_profile_data(general_data)
            except Exception as e:
                logger.warning(f"⚠️ Failed to fetch user profile: {str(e)}")

            # Step 2: Prepare calendar and time context for group detection
            prepare_calendar = PrepareCalendarData(
                external_api_service=self.external_api_service,
                user_id=user_id,
                provider_name=provider_name,
                timezone=timezone,
                current_dt=current_dt,
                user_profile=user_profile_data,
            )
            calendar_data = await prepare_calendar.prepare_all_calendar_data()
            events = calendar_data.get(CalendarDataConstants.KEY_EVENTS, [])
            calendar_metrics = calendar_data.get(
                CalendarDataConstants.KEY_CALENDAR_METRICS, {}
            )

            prepare_time = PrepareTimeData(
                user_profile=user_profile_data,
                current_dt=current_dt,
                tz=tz,
                timezone=timezone,
                calendar_events=events or [],
            )
            time_data = prepare_time.prepare_all_time_data()

            # Step 3: Prepare health parameters
            health_params: Dict[str, Any] = {}
            if self.external_api_service:
                prepare_health = PrepareHealthData(
                    external_api_service=self.external_api_service,
                    user_id=user_id,
                    user_profile=user_profile_data,
                    timezone=timezone,
                    time_data=time_data,
                )
                health_params = await prepare_health.prepare_all_health_params()

            latest_mood: Dict[str, Any] = {}
            try:
                # Always call API for fresh mood data
                if self.external_api_service:
                    mood_response = await self.external_api_service.get_latest_mood(
                        user_id=user_id
                    )
                    if mood_response:
                        mood_dto = MoodDTO(**mood_response)
                        latest_mood = mood_dto.model_dump(by_alias=True)
            except Exception as e:
                logger.warning(f"⚠️ Failed to fetch mood data: {str(e)}")

            fresh_mood, mood_not_logged_today = self._extract_fresh_mood(
                latest_mood, time_data
            )
            if fresh_mood:
                latest_mood = {**latest_mood, "not_logged_today": False}
            else:
                latest_mood = {"not_logged_today": mood_not_logged_today}
            logger.info(
                f"🧭 Mood freshness (health): fresh_mood={fresh_mood}, "
                f"mood_not_logged_today={mood_not_logged_today}"
            )

            # Step 5: Detect group similar to productivity flow
            group = self._detect_health_insight_group(
                calendar_events=events or [],
                health_params=health_params,
                user_profile=user_profile_data,
                time_data=time_data,
                calendar_metrics=calendar_metrics,
            )
            logger.info(
                f"🏥 Health insight group={group} current_hour="
                f"{time_data.get(TimeDataConstants.KEY_CURRENT_HOUR) if time_data else None}"
            )

            # Load cross-insight context to keep consistency with prior insights
            overall_context = ""
            productivity_context = ""
            try:
                overall_key = self._get_cache_key(
                    user_id, user_language, insight_type="overall"
                )
                overall_cached = await self.redis_client.get_data(overall_key)
                if overall_cached:
                    overall_context = str(
                        overall_cached.get(InsightGroupConstants.KEY_INSIGHT, "")
                    )
            except Exception as e:
                logger.warning(f"⚠️ Failed to load overall context: {str(e)}")

            try:
                productivity_key = self._get_cache_key(
                    user_id, user_language, insight_type="productivity"
                )
                productivity_cached = await self.redis_client.get_data(productivity_key)
                if productivity_cached:
                    productivity_context = str(
                        productivity_cached.get(InsightGroupConstants.KEY_INSIGHT, "")
                    )
            except Exception as e:
                logger.warning(f"⚠️ Failed to load productivity context: {str(e)}")

            # Build shared day context for cross-insight coherence
            day_context = await self._build_day_context(
                user_id=user_id,
                health_params=health_params,
                time_data=time_data,
                calendar_metrics=calendar_metrics,
                calendar_events=events or [],
                language=gen_language,
            )

            current_time = (
                time_data.get(TimeDataConstants.KEY_CURRENT_TIME_ISO)
                or current_dt.isoformat()
            )

            # Step 6: Build prompt & call LLM
            system_prompt, user_prompt = get_health_insight_prompt(
                group=group,
                health_params=health_params,
                latest_mood=latest_mood,
                user_profile=user_profile_data,
                time_data=time_data,
                calendar_metrics=calendar_metrics,
                current_time=current_time,
                timezone=timezone,
                language=gen_language,
                overall_context=overall_context,
                productivity_context=productivity_context,
            )

            if day_context:
                system_prompt = (
                    system_prompt
                    + "\n\n**Day Profile (shared context):**\n"
                    + day_context
                )
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
            response = await self.llm.ainvoke(messages)

            content_text = llm_response_text(response)
            parsed_insight = self._parse_json_response(content_text)
            parsed_insight = await translate_insight_dict(
                parsed_insight,
                user_language,
                self.llm,
                insight_type="health_insight",
            )

            # Force Vietnamese wording consistency for wind-down terms
            for key, value in parsed_insight.items():
                if isinstance(value, str):
                    parsed_insight[key] = self._normalize_wind_down_terms(
                        value, user_language
                    )

            insight_text = json.dumps(parsed_insight, ensure_ascii=False)

            return {
                "status": "success",
                "user_id": user_id,
                "insight": insight_text,
                "error": None,
            }
        except Exception as e:
            logger.error(
                f"❌ Error analyzing health insight for user {user_id}: {str(e)}"
            )
            return {
                "status": "error",
                "user_id": user_id,
                "insight": None,
                "error": f"Error during health insight analysis: {str(e)}",
            }

    async def analyze_overall_insight(
        self,
        user_id: str,
        timezone: Optional[str] = None,
        provider_name: Optional[str] = "",
        language: Optional[str] = "en-US",
        force_update: Optional[bool] = False,
    ) -> Dict[str, Any]:
        try:
            # Parse current time
            if timezone:
                tz = ZoneInfo(timezone)
                current_dt = datetime.now(tz)
            else:
                tz = ZoneInfo("UTC")
                current_dt = datetime.now(tz)

            user_language = language or "en-US"
            gen_language = resolve_generation_language(user_language)

            # Check cache first (unless force_update=True)
            cache_key = self._get_cache_key(
                user_id, user_language, insight_type="overall"
            )
            if not force_update:
                cached_result = await self.redis_client.get_data(cache_key)

                if cached_result:
                    logger.info(
                        f"✅ Found cached overall insight for user {user_id}, returning cached result"
                    )
                    return {
                        "status": "success",
                        "user_id": user_id,
                        "insight": cached_result.get(InsightGroupConstants.KEY_INSIGHT),
                        "error": None,
                    }
            else:
                logger.info(
                    f"🔄 force_update=True, bypassing overall insight cache for user {user_id}"
                )

            logger.info(
                f"🔍 Starting overall insight analysis for user {user_id} (timezone: {timezone}, provider: {provider_name})"
            )

            # Step 1: Get user profile data first (needed for calendar metrics)
            user_profile_data = None
            try:
                general_data = await CacheHelpers.fetch_general_user_data_cached(
                    self.redis_client,
                    self.external_api_service,
                    user_id,
                    log_prefix="Overall insight",
                )
                if general_data:
                    user_profile_data = self._extract_user_profile_data(general_data)
            except Exception as e:
                logger.warning(f"⚠️ Failed to fetch user profile: {str(e)}")

            # Step 2: Prepare calendar data (with user_profile for active hours)
            prepare_calendar = PrepareCalendarData(
                external_api_service=self.external_api_service,
                user_id=user_id,
                provider_name=provider_name,
                timezone=timezone,
                current_dt=current_dt,
                user_profile=user_profile_data,
            )
            calendar_data = await prepare_calendar.prepare_all_calendar_data()

            events = calendar_data.get(CalendarDataConstants.KEY_EVENTS, [])
            formatted_calendar_data = calendar_data.get(
                CalendarDataConstants.KEY_FORMATTED_CALENDAR_DATA, ""
            )
            calendar_metrics = calendar_data.get(
                CalendarDataConstants.KEY_CALENDAR_METRICS, {}
            )

            # Step 3: Prepare all time data using PrepareTimeData
            prepare_time = PrepareTimeData(
                user_profile=user_profile_data,
                timezone=timezone,
                current_dt=current_dt,
                tz=tz,
                calendar_events=events or [],
            )
            time_data = prepare_time.prepare_all_time_data()

            logger.info(f"📊 Preparing health data for user {user_id}")
            health_params = {}
            health_data = None
            try:
                if self.external_api_service:
                    prepare_health = PrepareHealthData(
                        external_api_service=self.external_api_service,
                        user_id=user_id,
                        user_profile=user_profile_data,
                        timezone=timezone,
                        time_data=time_data,
                    )

                    health_params = await prepare_health.prepare_all_health_params()
                    logger.info(f"✅ Successfully prepared all health parameters")

                    # Get health_data from cached data (already fetched in prepare_all_health_params)
                    health_data = prepare_health.get_health_data()
                    if health_data:
                        logger.info(
                            f"✅ Successfully retrieved health_data from cache for group detection"
                        )
                else:
                    logger.warning(
                        "⚠️ External API service not available for health data"
                    )
            except Exception as e:
                logger.warning(f"⚠️ Failed to prepare health data: {str(e)}")

            mood = None
            latest_mood: Optional[Dict[str, Any]] = None
            mood_not_logged_today = True
            try:
                mood_key = f"mood:{user_id}"
                latest_mood = await self.redis_client.get_data(mood_key)
                if latest_mood:
                    logger.info(f"✅ Retrieved mood from cache for user {user_id}")
                else:
                    # Always call API for fresh mood data
                    if self.external_api_service:
                        logger.info(
                            f"🔍 Mood cache miss, fetching from API for user {user_id}"
                        )
                        mood_response = await self.external_api_service.get_latest_mood(
                            user_id=user_id
                        )
                        if mood_response:
                            logger.info(
                                f"✅ Retrieved mood from API for user {user_id}"
                            )
                            # Parse response using MoodDTO model to extract only mood, notes, and moodDate
                            mood_dto = MoodDTO(**mood_response)
                            # Convert to dict with snake_case keys for response
                            latest_mood = mood_dto.model_dump(by_alias=True)
                            logger.info(f"✅ Retrieved mood from API for user {user_id}")
                    else:
                        logger.warning(
                            "⚠️ External API service not available for mood data"
                        )

                mood, mood_not_logged_today = self._extract_fresh_mood(
                    latest_mood, time_data
                )
                logger.info(
                    f"🧭 Mood freshness: mood={mood}, "
                    f"mood_not_logged_today={mood_not_logged_today}"
                )
            except Exception as e:
                logger.warning(f"⚠️ Failed to fetch mood data: {str(e)}")

            # Step 5: Detect which overall insight group applies
            group = self._detect_overall_insight_group(
                calendar_events=events or [],
                health_params=health_params,
                user_profile=user_profile_data,
                time_data=time_data,
                calendar_metrics=calendar_metrics,
                mood=mood,
            )
            logger.info(f"🎯 Detected overall insight group: {group}")

            # Step 6: Get appropriate prompt based on group
            # Extract user profile targets for prompts
            targets = (
                user_profile_data.get(TargetKeys.TARGETS, {})
                if user_profile_data
                else {}
            )
            bedtime_start = user_profile_data.get(TargetKeys.BEDTIME_START)
            bedtime_end = user_profile_data.get(TargetKeys.BEDTIME_END)
            active_start = user_profile_data.get(TargetKeys.ACTIVE_HOURS_START_TIME)
            active_end = user_profile_data.get(TargetKeys.ACTIVE_HOURS_END_TIME)
            break_interval = (
                targets.get(TargetKeys.BREAK_INTERVAL_HOURS, {}).get(TargetKeys.VALUE)
                if isinstance(targets.get(TargetKeys.BREAK_INTERVAL_HOURS, {}), dict)
                else None
            )
            break_duration = (
                targets.get(TargetKeys.BREAK_DURATION_MIN, {}).get(TargetKeys.VALUE)
                if isinstance(targets.get(TargetKeys.BREAK_DURATION_MIN, {}), dict)
                else None
            )

            # Extract health data values for prompts from health_params
            steps_today = health_params.get(HealthDataConstants.KEY_STEPS_TODAY)
            steps_goal = health_params.get(HealthDataConstants.KEY_STEPS_GOAL)
            sleep_lastnight = health_params.get(HealthDataConstants.KEY_SLEEP_LASTNIGHT)
            sleep_goal = health_params.get(HealthDataConstants.KEY_SLEEP_GOAL)
            latest_heart_rate = health_params.get(
                HealthDataConstants.KEY_LATEST_HEART_RATE
            )
            resting_heart_rate = health_params.get(
                HealthDataConstants.KEY_RESTING_HEART_RATE
            )
            sleep_quality = health_params.get(HealthDataConstants.KEY_SLEEP_QUALITY)
            baseline_resting_hr = health_params.get(
                HealthDataConstants.KEY_BASELINE_RESTING_HR
            )
            bedtime_streak = health_params.get(HealthDataConstants.KEY_BEDTIME_STREAK)
            sleep_last3nights = health_params.get(
                HealthDataConstants.KEY_SLEEP_LAST3NIGHTS
            )
            steps_streak = health_params.get(HealthDataConstants.KEY_STEPS_STREAK)
            weekly_health_progress = health_params.get(
                HealthDataConstants.KEY_WEEKLY_HEALTH_PROGRESS
            )
            daily_health_progress = health_params.get(
                HealthDataConstants.KEY_DAILY_HEALTH_PROGRESS
            )
            wind_down_buffer_mins = health_params.get(
                HealthDataConstants.KEY_WIND_DOWN_BUFFER_MINS
            )
            last_wake_time = health_params.get(HealthDataConstants.KEY_LAST_WAKE_TIME)
            first_sleep_time = health_params.get(
                HealthDataConstants.KEY_FIRST_SLEEP_TIME
            )

            # Extract calculated values from time_data (gộp time_context và time_metrics)
            time_to_next_event = time_data.get(TimeDataConstants.KEY_TIME_TO_NEXT_EVENT)
            free_slot_length = time_data.get(TimeDataConstants.KEY_FREE_SLOT_LENGTH)
            slot_length = free_slot_length
            time_to_bedtime = time_data.get(TimeDataConstants.KEY_TIME_TO_BEDTIME)
            in_active_window = time_data.get(TimeDataConstants.KEY_IN_ACTIVE_WINDOW)
            current_time = (
                time_data.get(TimeDataConstants.KEY_CURRENT_TIME_ISO)
                or current_dt.isoformat()
            )
            is_weekend = time_data.get(TimeDataConstants.KEY_IS_WEEKEND, False)
            current_hour = time_data.get(TimeDataConstants.KEY_CURRENT_HOUR)

            # Extract calendar metrics
            calendar_density = calendar_metrics.get(
                CalendarDataConstants.KEY_CALENDAR_DENSITY
            )
            back_to_back_count = calendar_metrics.get(
                CalendarDataConstants.KEY_BACK_TO_BACK_COUNT
            )
            meeting_minutes = calendar_metrics.get(
                CalendarDataConstants.KEY_MEETING_MINUTES
            )
            continuous_events_minutes = calendar_metrics.get(
                CalendarDataConstants.KEY_CONTINUOUS_EVENTS_MINUTES
            )
            work_events_hours_weekend = calendar_metrics.get(
                CalendarDataConstants.KEY_WORK_EVENTS_HOURS_WEEKEND
            )
            work_load_high = calendar_metrics.get(
                CalendarDataConstants.KEY_WORK_LOAD_HIGH
            )

            # Extract calendar events for different time windows
            calendar_events_next7d = calendar_data.get(
                CalendarDataConstants.KEY_CALENDAR_EVENTS_NEXT7D, []
            )
            calendar_events_next48h = calendar_data.get(
                CalendarDataConstants.KEY_CALENDAR_EVENTS_NEXT48H, []
            )
            calendar_events_next3h = calendar_data.get(
                CalendarDataConstants.KEY_CALENDAR_EVENTS_NEXT3H, []
            )

            productivity_key = self._get_cache_key(
                user_id,
                user_language,
                insight_type=InsightGroupConstants.KEY_PRODUCTIVITY_TYPE,
            )
            productivity_insight = await self.redis_client.get_data(productivity_key)
            productivity_context = ""
            if productivity_insight:
                productivity_context = productivity_insight.get(
                    InsightGroupConstants.KEY_INSIGHT
                )
            logger.info(
                f"sleep_quality: {sleep_quality}, sleep_goal: {sleep_goal}, sleep_lastnight: {sleep_lastnight}"
            )

            # Build shared day context for cross-insight coherence
            day_context = await self._build_day_context(
                user_id=user_id,
                health_params=health_params,
                time_data=time_data,
                calendar_metrics=calendar_metrics,
                calendar_events=events or [],
                language=gen_language,
            )

            # Reminders merged into calendar_data for group prompts that render
            # a Calendar Events section; groups without calendar ignore it.
            reminders_today = await self._fetch_today_reminders(
                user_id=user_id,
                timezone=timezone,
                current_dt=current_dt,
            )
            reminders_section = self._build_reminders_section(reminders_today, timezone)
            calendar_for_prompt = self._merge_calendar_with_reminders(
                formatted_calendar_data, reminders_section
            )

            system_prompt, user_prompt = get_overall_insight_prompt(
                group=group,
                calendar_data=calendar_for_prompt,
                health_data=health_data,
                user_profile=user_profile_data,
                current_time=current_time,
                timezone=timezone,
                steps_today=steps_today,
                steps_goal=steps_goal,
                sleep_lastnight=sleep_lastnight,
                sleep_goal=sleep_goal,
                sleep_quality=sleep_quality,
                latest_heart_rate=latest_heart_rate,
                resting_heart_rate=resting_heart_rate,
                baseline_resting_hr=baseline_resting_hr,
                bedtime_streak=bedtime_streak,
                sleep_last3nights=sleep_last3nights,
                steps_streak=steps_streak,
                weekly_health_progress=weekly_health_progress,
                daily_health_progress=daily_health_progress,
                wind_down_buffer_mins=wind_down_buffer_mins,
                time_to_next_event=time_to_next_event,
                free_slot_length=free_slot_length,
                time_to_bedtime=time_to_bedtime,
                in_active_window=in_active_window,
                is_weekend=is_weekend,
                current_hour=current_hour,
                bedtime_start=bedtime_start,
                bedtime_end=bedtime_end,
                active_hours_start=active_start,
                active_hours_end=active_end,
                calendar_density=calendar_density,
                back_to_back_count=back_to_back_count,
                meeting_minutes=meeting_minutes,
                continuous_events_minutes=continuous_events_minutes,
                work_events_hours_weekend=work_events_hours_weekend,
                work_load_high=work_load_high,
                calendar_events_next7d=calendar_events_next7d,
                calendar_events_next48h=calendar_events_next48h,
                calendar_events_next3h=calendar_events_next3h,
                mood=mood,  # mood was fetched before group detection
                mood_not_logged_today=mood_not_logged_today,
                last_wake_time=last_wake_time,
                first_sleep_time=first_sleep_time,
                language=gen_language,
                productivity_context=productivity_context,
            )

            if day_context:
                system_prompt = (
                    system_prompt
                    + "\n\n**Day Profile (shared context):**\n"
                    + day_context
                )
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]

            response = await self.llm.ainvoke(messages)

            # Extract text from response and parse JSON
            content_text = llm_response_text(response)
            parsed_insight = self._parse_json_response(content_text)

            # Validate that parsed_insight has the expected structure
            if not isinstance(parsed_insight, dict):
                raise ValueError("LLM response is not a valid JSON object")

            cross_module_great_job = None
            cross_module_need_attention = None
            evening_overall_group = group in self._EVENING_OVERALL_INSIGHT_GROUPS
            try:
                time_window_pass = self._is_evening_window_for_cross_module(current_dt)
                random_pass = False
                cross_sections: Dict[str, str] = {
                    "great_job": "",
                    "need_attention": "",
                }

                # Skip cross-module replacement during evening sleep/wind-down groups
                # to avoid finance/productivity praise conflicting with rest-focused insights.
                if time_window_pass and not evening_overall_group:
                    random_pass = self._should_random_replace_need_attention()

                if time_window_pass and random_pass and not evening_overall_group:
                    cross_inputs = await self._collect_cross_module_inputs(
                        user_id=user_id,
                        timezone=timezone,
                        current_dt=current_dt,
                        health_data=health_data,
                        health_params=health_params,
                        calendar_metrics=calendar_metrics,
                    )
                    cross_sections = await self._evaluate_cross_module_rules(
                        inputs=cross_inputs,
                        current_dt=current_dt,
                        language=user_language,
                    )

                has_great_job = bool(cross_sections.get("great_job", "").strip())
                has_need_attention = bool(
                    cross_sections.get("need_attention", "").strip()
                )
                logger.info(
                    f"🧠 Cross-Module Intelligence gate - time_window_pass={time_window_pass} random_pass={random_pass} has_great_job={has_great_job} has_need_attention={has_need_attention}"
                )
                if time_window_pass and random_pass and not evening_overall_group:
                    cm_gj = (cross_sections.get("great_job") or "").strip()
                    cm_na = (cross_sections.get("need_attention") or "").strip()
                    if cm_gj and not parsed_insight.get("great_job", "").strip():
                        cross_module_great_job = cm_gj
                    if cm_na and not parsed_insight.get("need_attention", "").strip():
                        cross_module_need_attention = cm_na
            except Exception as cross_e:
                logger.warning(
                    f"⚠️ Cross-module need_attention replacement skipped due to error: {str(cross_e)}"
                )

            result_insight = {
                "great_job": (
                    cross_module_great_job
                    if cross_module_great_job
                    else parsed_insight.get("great_job", "")
                ),
                "need_attention": (
                    cross_module_need_attention
                    if cross_module_need_attention
                    else parsed_insight.get("need_attention", "")
                ),
                "opportunity": parsed_insight.get("opportunity", ""),
            }
            result_insight = await translate_insight_dict(
                result_insight,
                user_language,
                self.llm,
                insight_type="overall_insight",
            )
            result_insight = {
                "great_job": self._normalize_wind_down_terms(
                    str(result_insight.get("great_job", "") or ""),
                    user_language,
                ),
                "need_attention": self._normalize_wind_down_terms(
                    str(result_insight.get("need_attention", "") or ""),
                    user_language,
                ),
                "opportunity": self._normalize_wind_down_terms(
                    str(result_insight.get("opportunity", "") or ""),
                    user_language,
                ),
            }
            result_insight = self._compact_overall_insight(result_insight)

            logger.info(f"✅ Successfully generated overall insight for user {user_id}")

            return {
                "status": "success",
                "user_id": user_id,
                "insight": result_insight,
                "error": None,
            }

        except Exception as e:
            logger.error(
                f"❌ Error analyzing overall insight for user {user_id}: {str(e)}"
            )
            import traceback

            logger.error(traceback.format_exc())
            return {
                "status": "error",
                "user_id": user_id,
                "insight": None,
                "error": f"Error during overall insight analysis: {str(e)}",
            }

    async def debug_pipeline(
        self,
        raw_data: Dict[str, Any],
        domain: str,
        language: str,
    ) -> Dict[str, Any]:
        """Debug the insight pipeline block-by-block, returning all intermediate states."""
        from insights.services.advanced_insight_service import AdvancedInsightService

        try:
            service = AdvancedInsightService(
                external_api_service=self.external_api_service
            )
            return await service.debug_pipeline(
                raw_data=raw_data, domain=domain, language=language
            )
        except Exception as e:
            logger.error(f"❌ debug_pipeline failed: {str(e)}")
            return {"status": "error", "error": str(e)}

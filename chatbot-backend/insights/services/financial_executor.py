from __future__ import annotations

import asyncio
import json
import os
from calendar import monthrange
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from agents.llm_helper import llm_response_text
from agents.llm_helper import parse_json_object_from_llm_text
from agents.llm_helper import resolve_generation_language
from agents.llm_helper import translate_insight_dict

from agents.llm_manager import LLMManager
from agents.prompt import get_financial_combined_insights_llm_prompts
from agents.prompt import get_language_name
from clients.redis_client import RedisClient
from insights.services.financial_base import FinancialInsightService
from langchain_core.messages import HumanMessage, SystemMessage
from services.executor.cache_helpers import CacheHelpers
from services.external_api_service import IExternalAPIService
from utils.logger import logger

_ZERO = 0.0


def _ym_str(year: int, month: int) -> str:
    return f"{year}-{month:02d}"


def _month_label(ym: str) -> str:
    try:
        dt = datetime.strptime(ym, "%Y-%m")
        return dt.strftime("%B %Y")
    except ValueError:
        return ym


def _day_of_month_fraction(ym: str) -> float:
    try:
        dt = datetime.strptime(ym, "%Y-%m")
        today = date.today()
        if today.year != dt.year or today.month != dt.month:
            return 1.0
        days_in_month = monthrange(dt.year, dt.month)[1]
        return min(today.day / days_in_month, 1.0)
    except ValueError:
        return 1.0


def _prev_months(ym: str, n: int) -> List[str]:
    dt = datetime.strptime(ym, "%Y-%m")
    result = []
    for i in range(n, 0, -1):
        m = dt.month - i
        y = dt.year
        while m <= 0:
            m += 12
            y -= 1
        result.append(_ym_str(y, m))
    return result


def _end_of_month_window(ym: str) -> bool:
    today = date.today()
    try:
        dt = datetime.strptime(ym, "%Y-%m").date()
    except ValueError:
        return False
    last_day = monthrange(dt.year, dt.month)[1]
    end_date = date(dt.year, dt.month, last_day)
    next_m = dt.month + 1
    next_y = dt.year
    if next_m > 12:
        next_m = 1
        next_y += 1
    follow_day2 = date(next_y, next_m, 2)
    return end_date <= today <= follow_day2


def _pct(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator * 100


def _d(val: Any, default: float = 0.0) -> float:
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def evaluate_insights(
    ym: str,
    summaries: Dict[str, Dict[str, Any]],  # month_str -> summary dict
    goals: List[Dict[str, Any]],
    logs: List[Dict[str, Any]],
    language: str,
) -> List[Dict[str, Any]]:
    """Apply all condition rules and return triggered insights."""

    insights: List[Dict[str, Any]] = []
    month_label = _month_label(ym)
    eom_window = _end_of_month_window(ym)
    fraction = _day_of_month_fraction(ym)

    cur = summaries.get(ym, {})
    cur_income = _d(cur.get("totalIncome"))
    cur_spend = _d(cur.get("totalSpending"))
    cur_savings = _d(cur.get("totalSavings"))
    cur_savings_rate = _d(cur.get("savingsRatePercentage"))
    cur_score = cur.get("financeScore")
    spending_breakdown: List[Dict[str, Any]] = cur.get("spendingBreakdown") or []

    prev_month_strs = _prev_months(ym, 6)
    prev1_ym = prev_month_strs[-1] if prev_month_strs else None
    prev = summaries.get(prev1_ym, {}) if prev1_ym else {}
    prev_income = _d(prev.get("totalIncome"))
    prev_savings = _d(prev.get("totalSavings"))
    prev_savings_rate = _d(prev.get("savingsRatePercentage"))
    prev_score = prev.get("financeScore")
    prev_spend = _d(prev.get("totalSpending"))

    # Budget from summary (backend calculates incomeTarget / spendingBudget via budget records)
    income_target = _d(cur.get("incomeTarget"))
    spending_budget = _d(cur.get("spendingBudget"))

    # Pro-rated expected values
    expected_income = income_target * fraction if income_target else None
    expected_spend = spending_budget * fraction if spending_budget else None

    def add(
        category: str, condition: str, urgency: str, message: str, detail: str = ""
    ) -> None:
        # Response exposes only `message`; merge optional detail into the text for rules/LLM input.
        _ = (category, condition, urgency)
        full = f"{message} ({detail})" if detail else message
        insights.append({"message": full})

    # =========================================================================
    # INCOME TRACKING
    # =========================================================================

    # 1. Income drops >15% vs previous month - end-of-month window
    if eom_window and prev_income > 0 and cur_income < prev_income * 0.85:
        drop_pct = round((1 - cur_income / prev_income) * 100)
        add(
            "Income Tracking",
            "income_drop_15pct",
            "now",
            f"Your income dropped by {drop_pct}% compared to last month.",
            f"Current: {cur_income:.2f}, Previous: {prev_income:.2f}",
        )

    # 2. Income < income target - end-of-month window
    if eom_window and income_target > 0 and cur_income < income_target:
        add(
            "Income Tracking",
            "income_below_target",
            "now",
            f"Your income fell short of the target for {month_label}.",
            f"Actual: {cur_income:.2f}, Target: {income_target:.2f}",
        )

    # 3. Cumulative income < pro-rated income by >20% - any day
    if (
        expected_income is not None
        and expected_income > 0
        and cur_income < expected_income * 1.20
    ):
        under_pct = round((1 - cur_income / expected_income) * 100)
        add(
            "Income Tracking",
            "income_under_prorated_20pct",
            "now",
            f"Your income is {under_pct}% under the expected amount so far this month.",
            f"Actual: {cur_income:.2f}, Expected so far: {expected_income:.2f}",
        )

    # 4. Income increases >10% vs last 3-month average
    prev3 = [_d(summaries.get(m, {}).get("totalIncome")) for m in prev_month_strs[-3:]]
    avg3 = sum(prev3) / len(prev3) if prev3 else 0
    if eom_window and avg3 > 0 and cur_income > avg3 * 1.10:
        inc_pct = round((cur_income / avg3 - 1) * 100)
        add(
            "Income Tracking",
            "income_up_10pct_3m_avg",
            "positive",
            f"Your income increased by {inc_pct}% over the last-quarter average.",
            f"Current: {cur_income:.2f}, 3-month avg: {avg3:.2f}",
        )

    # =========================================================================
    # INCOME STABILITY
    # =========================================================================
    incomes6 = [_d(summaries.get(m, {}).get("totalIncome")) for m in prev_month_strs]
    if len(incomes6) >= 3:
        avg6 = sum(incomes6) / len(incomes6)
        if avg6 > 0:
            variance = sum((x - avg6) ** 2 for x in incomes6) / len(incomes6)
            volatility = (variance**0.5) / avg6
            if volatility > 0.30:
                add(
                    "Income Stability",
                    "income_volatility_high",
                    "now",
                    "Your income fluctuates significantly. Consider building a 6-month emergency fund.",
                    f"Volatility coefficient: {volatility:.0%}",
                )

    # =========================================================================
    # SPENDING MONITORING
    # =========================================================================

    # 5. Spending > budget - end-of-month window
    if eom_window and spending_budget > 0 and cur_spend > spending_budget:
        over_pct = round((cur_spend / spending_budget - 1) * 100)
        add(
            "Spending Monitoring",
            "spending_over_budget",
            "now",
            f"Your spending exceeded the budget by {over_pct}% in {month_label}.",
            f"Spent: {cur_spend:.2f}, Budget: {spending_budget:.2f}",
        )

    # 6. Cumulative spending > pro-rated budget by >20% - any day
    if (
        expected_spend is not None
        and expected_spend > 0
        and cur_spend > expected_spend * 1.20
    ):
        over_pct = round((cur_spend / expected_spend - 1) * 100)
        add(
            "Spending Monitoring",
            "spending_over_prorated_20pct",
            "now",
            f"Your spending is {over_pct}% above the expected budget so far this month.",
            f"Spent: {cur_spend:.2f}, Expected so far: {expected_spend:.2f}",
        )

    # 7. Spending > income - any day
    if cur_income > 0 and cur_spend > cur_income:
        add(
            "Spending Monitoring",
            "spending_exceeds_income",
            "now",
            "Your expenses exceeded your income so far this month.",
            f"Income: {cur_income:.2f}, Spending: {cur_spend:.2f}",
        )

    # 8. Category spending increases >30% month-to-month
    if eom_window and spending_breakdown and prev:
        prev_breakdown: List[Dict[str, Any]] = prev.get("spendingBreakdown") or []
        prev_by_cat: Dict[str, float] = {
            item["category"]: _d(item.get("amount")) for item in prev_breakdown
        }
        for item in spending_breakdown:
            cat = item.get("category", "")
            cat_display = item.get("categoryDisplayName", cat)
            cur_cat_amt = _d(item.get("amount"))
            prev_cat_amt = prev_by_cat.get(cat, 0)
            if prev_cat_amt > 0 and cur_cat_amt > prev_cat_amt * 1.30:
                spike_pct = round((cur_cat_amt / prev_cat_amt - 1) * 100)
                add(
                    "Spending Monitoring",
                    "category_spend_spike_30pct",
                    "now",
                    f"Your {cat_display} spending increased by {spike_pct}% compared to last month.",
                    f"Current: {cur_cat_amt:.2f}, Previous: {prev_cat_amt:.2f}",
                )

    # Spending grows faster than income
    if eom_window and prev_income > 0 and prev_spend > 0:
        income_growth = (cur_income - prev_income) / prev_income
        spend_growth = (cur_spend - prev_spend) / prev_spend
        if spend_growth > income_growth:
            add(
                "Behavioural Insights",
                "spending_growth_faster_than_income",
                "now",
                "Your spending is increasing faster than your income.",
                f"Income growth: {income_growth:.1%}, Spending growth: {spend_growth:.1%}",
            )

    # =========================================================================
    # SAVINGS BEHAVIOUR
    # =========================================================================

    # Find savings target from goals (type SAVINGS or name contains saving/savings)
    savings_target = 0.0
    for goal in goals:
        if (
            "sav" in (goal.get("goalType") or "").lower()
            or "sav" in (goal.get("name") or "").lower()
        ):
            savings_target += _d(goal.get("contributionAmount"))

    # 9. Savings < savings target - end-of-month window
    if eom_window and savings_target > 0 and cur_savings < savings_target:
        today = date.today()
        dt = datetime.strptime(ym, "%Y-%m").date()
        if today.month == dt.month and today.year == dt.year:
            tail = "You should consider better planning for the month ahead."
        else:
            next_m = dt.month + 1
            next_y = dt.year
            if next_m > 12:
                next_m = 1
                next_y += 1
            follow_1st = date(next_y, next_m, 1)
            follow_2nd = date(next_y, next_m, 2)
            if follow_1st <= today <= follow_2nd:
                tail = "Let's reset this month!"
            else:
                tail = "You should consider better planning for the month ahead."
        add(
            "Savings Behaviour",
            "savings_below_target",
            "now",
            f"Your savings fell short of the target for {month_label}. {tail}",
            f"Actual savings: {cur_savings:.2f}, Target: {savings_target:.2f}",
        )

    # 10. Savings rate < 10% - end-of-month window
    if eom_window and cur_income > 0 and cur_savings_rate < 10:
        add(
            "Savings Behaviour",
            "savings_rate_below_10pct",
            "now",
            f"Your savings rate is below the recommended 10% in {month_label}.",
            f"Savings rate: {cur_savings_rate}%",
        )

    # 11. Savings rate > 20% - end-of-month window (positive)
    if eom_window and cur_income > 0 and cur_savings_rate > 20:
        add(
            "Savings Behaviour",
            "savings_rate_above_20pct",
            "positive",
            f"Excellent — you saved more than 20% of your income in {month_label}!",
            f"Savings rate: {cur_savings_rate}%",
        )

    # 12. Savings rate improving month-to-month - end-of-month window
    if eom_window and prev_savings_rate > 0 and cur_savings_rate > prev_savings_rate:
        improvement = cur_savings_rate - prev_savings_rate
        add(
            "Savings Behaviour",
            "savings_rate_improving",
            "positive",
            f"Your savings rate improved by {improvement:.1f}% in {month_label}. Keep it up!",
            f"Current: {cur_savings_rate}%, Previous: {prev_savings_rate}%",
        )

    # 13. Savings increases for 3 consecutive months
    if eom_window and len(prev_month_strs) >= 3:
        s3 = [
            _d(summaries.get(m, {}).get("savingsRatePercentage"))
            for m in prev_month_strs[-3:]
        ]
        if s3[0] < s3[1] < s3[2] < cur_savings_rate:
            add(
                "Savings Behaviour",
                "savings_improving_3_consecutive",
                "positive",
                "Great job — you've increased your savings consistently for 3 months in a row!",
            )

    # 14. Emergency fund < 3 months of expenses
    if eom_window and cur_spend > 0:
        months_covered = cur_savings / cur_spend if cur_spend > 0 else 0
        if months_covered < 3:
            add(
                "Savings Behaviour",
                "emergency_fund_low",
                "now",
                f"Your emergency fund currently covers only {months_covered:.1f} months of expenses.",
                f"Savings: {cur_savings:.2f}, Monthly spend: {cur_spend:.2f}",
            )

    # Consistent budget adherence 3 months
    if eom_window and len(prev_month_strs) >= 3:
        adherent_months = 0
        for m in prev_month_strs[-3:]:
            m_spend = _d(summaries.get(m, {}).get("totalSpending"))
            m_budget = _d(summaries.get(m, {}).get("spendingBudget"))
            if m_budget > 0 and m_spend <= m_budget:
                adherent_months += 1
        if adherent_months == 3:
            add(
                "Behavioural Insights",
                "consistent_budget_adherence_3m",
                "positive",
                "Great job staying within budget for 3 consecutive months!",
            )

    # =========================================================================
    # FINANCIAL GOALS
    # =========================================================================
    today_date = date.today()
    for goal in goals:
        goal_name = goal.get("name", "your goal")
        target_amount = _d(goal.get("targetAmount"))
        current_amount = _d(goal.get("currentAmount"))
        target_date_str = goal.get("targetDate")
        contribution_amount = _d(goal.get("contributionAmount"))
        progress_pct = _d(goal.get("progressPercentage"))

        # 15. Savings pace insufficient - end-of-month window
        if (
            eom_window
            and target_date_str
            and target_amount > 0
            and contribution_amount > 0
        ):
            try:
                target_date = date.fromisoformat(target_date_str)
                remaining = target_amount - current_amount
                months_left = (
                    (target_date.year - today_date.year) * 12
                    + target_date.month
                    - today_date.month
                )
                if months_left > 0:
                    required_monthly = remaining / months_left
                    if contribution_amount < required_monthly:
                        add(
                            "Financial Goals",
                            "goal_pace_insufficient",
                            "now",
                            f"You may miss your '{goal_name}' goal deadline at the current saving rate.",
                            f"Need {required_monthly:.2f}/month, contributing {contribution_amount:.2f}/month",
                        )
            except (ValueError, TypeError):
                pass

        # 16. Milestone reached (25%, 50%, 75%, 100%)
        if eom_window:
            for milestone in [25, 50, 75, 100]:
                if abs(progress_pct - milestone) < 1.5:
                    add(
                        "Financial Goals",
                        f"goal_milestone_{milestone}pct",
                        "positive",
                        f"You've reached {milestone}% of your '{goal_name}' goal!",
                        f"Saved: {current_amount:.2f} of {target_amount:.2f}",
                    )
                    break

    # 17. No contribution to goal for 60 days (check log timestamps)
    for goal in goals:
        goal_id = str(goal.get("id", ""))
        goal_name = goal.get("name", "your goal")
        last_contribution: Optional[date] = None
        for log in logs:
            if str(log.get("goalId", "")) == goal_id:
                log_date_str = log.get("logDate", "")
                try:
                    log_date = date.fromisoformat(log_date_str[:10])
                    if last_contribution is None or log_date > last_contribution:
                        last_contribution = log_date
                except (ValueError, TypeError):
                    pass
        if last_contribution is not None:
            days_since = (today_date - last_contribution).days
            if days_since > 60:
                add(
                    "Financial Goals",
                    "goal_no_contribution_60d",
                    "now",
                    f"You haven't contributed to your '{goal_name}' goal in {days_since} days.",
                )

    # =========================================================================
    # FINANCIAL HEALTH SCORE
    # =========================================================================

    # 18. Score drops > 10 points - end-of-month window
    if eom_window and cur_score is not None and prev_score is not None:
        cur_s = int(cur_score)
        prev_s = int(prev_score)
        if cur_s < prev_s - 10:
            drop = prev_s - cur_s
            add(
                "Financial Health Score",
                "score_declined_10pts",
                "now",
                f"Your financial health score declined by {drop} points in {month_label}.",
                f"Current score: {cur_s}, Previous score: {prev_s}",
            )
        elif cur_s > prev_s:
            improvement = cur_s - prev_s
            add(
                "Financial Health Score",
                "score_improved",
                "positive",
                f"Your financial health score improved by {improvement} points in {month_label}!",
                f"Current: {cur_s}, Previous: {prev_s}",
            )

    # =========================================================================
    # ANOMALY DETECTION (from logs)
    # =========================================================================
    total_net = _d(cur.get("totalNetWorth"))
    if cur_savings > 0 and total_net > 0 and cur_savings / total_net > 0.20:
        add(
            "Savings Behaviour",
            "large_balance_low_yield",
            "opportunity",
            "You may benefit from investing part of your idle savings.",
            f"Cash savings are {_pct(cur_savings, total_net):.0f}% of net worth",
        )

    return insights


# --- LLM synthesis (2 insights: month overview + aggregated rules) -----------------


def _compact_month_summary_for_llm(ym: str, cur: Dict[str, Any]) -> Dict[str, Any]:
    """Pick fields useful for a factual month recap prompt."""
    keys = [
        "totalIncome",
        "totalSpending",
        "totalSavings",
        "savingsRatePercentage",
        "financeScore",
        "incomeTarget",
        "spendingBudget",
        "totalNetWorth",
        "spendingBreakdown",
    ]
    out: Dict[str, Any] = {"month": ym, "monthLabel": _month_label(ym)}
    for k in keys:
        if k in cur and cur.get(k) is not None:
            out[k] = cur.get(k)
    return out


def _condense_raw_for_llm(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Messages only, for merge prompt."""
    return [{"message": r.get("message")} for r in raw if r.get("message")]


def _numeric_nonzero(value: Any) -> bool:
    """True if value represents a non-zero amount or score (strings parsed loosely)."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        s = value.strip().replace(",", "")
        if not s:
            return False
        try:
            return float(s) != 0.0
        except ValueError:
            return True
    return True


def _month_payload_has_meaningful_finance(month_payload: Dict[str, Any]) -> bool:
    """True when summary has non-zero metrics or a breakdown with non-zero amounts."""
    keys = (
        "totalIncome",
        "totalSpending",
        "totalSavings",
        "savingsRatePercentage",
        "financeScore",
        "incomeTarget",
        "spendingBudget",
        "totalNetWorth",
    )
    for k in keys:
        if _numeric_nonzero(month_payload.get(k)):
            return True
    bd = month_payload.get("spendingBreakdown")
    if isinstance(bd, list):
        for row in bd:
            if isinstance(row, dict):
                for v in row.values():
                    if _numeric_nonzero(v):
                        return True
            elif _numeric_nonzero(row):
                return True
        return False
    if isinstance(bd, dict):
        for v in bd.values():
            if _numeric_nonzero(v):
                return True
        return False
    return False


def _finance_inputs_indicate_data(
    month_payload: Dict[str, Any], raw_insights: List[Dict[str, Any]]
) -> bool:
    """True if rules fired or month summary has non-trivial numbers/breakdown (not all zeros)."""
    if raw_insights:
        return True
    return _month_payload_has_meaningful_finance(month_payload)


def _month_overview_and_merged_from_parsed(
    parsed: Optional[Dict[str, Any]], fallback_text: str
) -> tuple[str, str]:
    """Parse combined LLM JSON: month_overview + merged_signals."""
    if not parsed:
        return (fallback_text or "").strip(), ""
    mo = parsed.get("month_overview") or parsed.get("monthOverview")
    merged = parsed.get("merged_signals") or parsed.get("mergedSignals")
    m1 = (
        str(mo).strip()
        if mo is not None and str(mo).strip()
        else (fallback_text or "").strip()
    )
    m2 = str(merged).strip() if merged is not None else ""
    return m1, m2


def _resolve_has_data_flag(
    month_payload: Dict[str, Any],
    raw_insights: List[Dict[str, Any]],
) -> bool:
    """True only when rules fired or month summary has non-zero / meaningful breakdown."""
    return _finance_inputs_indicate_data(month_payload, raw_insights)


class FinancialInsightServiceImpl(FinancialInsightService):
    """Implementation for financial insights with external API source."""

    def __init__(self, external_api_service: IExternalAPIService):
        self.external_api_service = external_api_service
        self.redis_client = RedisClient()
        self._llm_manager: Optional[LLMManager] = None
        self._llm: Optional[Any] = None
        self._cache_ttl = 300

    def _ensure_llm(self) -> None:
        if self._llm_manager is None:
            provider = os.getenv("FINANCIAL_INSIGHT_LLM_PROVIDER", "bedrock")
            self._llm_manager = LLMManager(provider=provider)
            self._llm = self._llm_manager.create_model(temperature=0.1)

    async def _build_two_insights_with_llm(
        self,
        ym: str,
        language: str,
        summaries: Dict[str, Dict[str, Any]],
        raw_insights: List[Dict[str, Any]],
        prefer_currency: str = "",
        target_language: Optional[str] = None,
    ) -> tuple[List[Dict[str, Any]], bool]:
        self._ensure_llm()
        if self._llm is None:
            raise RuntimeError("LLM not initialized")

        cur = summaries.get(ym, {})
        month_payload = _compact_month_summary_for_llm(ym, cur)
        month_json = json.dumps(month_payload, ensure_ascii=False, default=str)
        signals = _condense_raw_for_llm(raw_insights)
        signals_json = json.dumps(signals, ensure_ascii=False, default=str)
        language_name = get_language_name(language)

        sys_p, user_p = get_financial_combined_insights_llm_prompts(
            language_name, ym, month_json, signals_json, prefer_currency=prefer_currency
        )
        resp = await self._llm.ainvoke(
            [SystemMessage(content=sys_p), HumanMessage(content=user_p)]
        )
        text = llm_response_text(resp)
        parsed = parse_json_object_from_llm_text(text)
        msg1, msg2 = _month_overview_and_merged_from_parsed(parsed, text)
        has_data = _resolve_has_data_flag(month_payload, raw_insights)
        _msg1 = msg1
        _msg2 = msg2

        localize_lang = target_language if target_language is not None else language
        translated = await translate_insight_dict(
            {"month_overview": _msg1, "merged_signals": _msg2},
            localize_lang,
            self._llm,
        )
        msg1 = str(translated.get("month_overview") or _msg1)
        msg2 = str(translated.get("merged_signals") or _msg2)

        insight1: Dict[str, Any] = {"message": msg1}
        if not raw_insights:
            return [insight1], has_data

        if not msg2.strip() and raw_insights:
            msg2 = str(raw_insights[0].get("message") or "").strip()

        insight2: Dict[str, Any] = {"message": msg2}
        return [insight1, insight2], has_data

    async def get_financial_insights(
        self,
        user_id: str,
        language: Optional[str] = "en-US",
        month: Optional[str] = None,
        force_update: bool = False,
    ) -> Dict[str, Any]:
        user_language = language or "en-US"
        gen_language = resolve_generation_language(user_language)
        ym = month if month else _ym_str(date.today().year, date.today().month)
        logger.info(
            f"💰 Financial insight request — user: {user_id}, month: {ym}, "
            f"force_update={force_update} - language: {language}"
        )
        try:
            # Light optimization: authenticate once before concurrent fan-out calls.
            token = await self.external_api_service.login_and_get_token()
            if not token:
                return {
                    "status": "error",
                    "user_id": user_id,
                    "month": ym,
                    "insights": [],
                    "has_data": False,
                    "error": "Financial service unavailable — could not authenticate with backend",
                }

            insight_cache_key = CacheHelpers.financial_insight_redis_key(
                user_id, user_language
            )
            if not force_update:
                try:
                    cached = await self.redis_client.get_data(insight_cache_key)
                    if isinstance(cached, dict) and isinstance(
                        cached.get("insights"), list
                    ):
                        valid = all(
                            isinstance(x, dict) and isinstance(x.get("message"), str)
                            for x in cached["insights"]
                        )
                        if valid:
                            logger.info(
                                f"✅ Financial insight cache hit — user: {user_id}, month: {ym}"
                            )
                            cached_has = cached.get("has_data")
                            has_data_flag = (
                                cached_has if isinstance(cached_has, bool) else True
                            )
                            return {
                                "status": "success",
                                "user_id": user_id,
                                "month": ym,
                                "insights": cached["insights"],
                                "has_data": has_data_flag,
                                "error": None,
                            }
                except Exception as exc:
                    logger.warning(
                        f"⚠️ Financial insight cache read failed for user {user_id}: {exc}"
                    )
            else:
                logger.info(
                    f"🔄 force_update=True, bypassing financial insight cache for user {user_id}"
                )

            months_to_fetch = _prev_months(ym, 6) + [ym]
            summary_results = await asyncio.gather(
                *[
                    self.external_api_service.get_finance_summary(user_id, m)
                    for m in months_to_fetch
                ],
                return_exceptions=True,
            )
            summaries: Dict[str, Dict[str, Any]] = {}
            for m, result in zip(months_to_fetch, summary_results):
                if isinstance(result, dict):
                    summaries[m] = result

            goals_task = self.external_api_service.get_finance_goals(user_id)
            logs_task = self.external_api_service.get_finance_logs(
                user_id, page=1, size=200
            )
            profile_task = CacheHelpers.fetch_general_user_data_cached(
                self.redis_client,
                self.external_api_service,
                user_id,
                log_prefix="Financial insights",
            )
            goals, logs, user_data = await asyncio.gather(
                goals_task, logs_task, profile_task
            )
            prefer_currency = ""
            if isinstance(user_data, dict):
                prefer_currency = str(
                    user_data.get("prefer_currency")
                    or user_data.get("preferCurrency")
                    or ""
                ).strip()

            raw_insights = evaluate_insights(
                ym=ym,
                summaries=summaries,
                goals=goals,
                logs=logs,
                language=language,
            )
            logger.info(
                f"✅ Financial rule evaluation complete — {len(raw_insights)} raw insight(s) for user {user_id}"
            )
            try:
                insights, has_data_flag = await self._build_two_insights_with_llm(
                    ym=ym,
                    language=gen_language,
                    summaries=summaries,
                    raw_insights=raw_insights,
                    prefer_currency=prefer_currency,
                    target_language=user_language,
                )
                logger.info(
                    f"✅ LLM synthesis complete — {len(insights)} insight(s) for user {user_id}"
                )
            except Exception as llm_exc:
                logger.warning(
                    f"⚠️ LLM financial insight synthesis failed, using rule-only insights: {llm_exc}"
                )
                insights = raw_insights
                month_payload_fb = _compact_month_summary_for_llm(
                    ym, summaries.get(ym, {})
                )
                has_data_flag = _finance_inputs_indicate_data(
                    month_payload_fb, raw_insights
                )

            try:
                await self.redis_client.set_data(
                    insight_cache_key,
                    {"insights": insights, "has_data": has_data_flag},
                    expire=self._cache_ttl,
                )
                logger.info(
                    f"💾 Cached financial insights for user {user_id}, month {ym} "
                    f"(TTL {self._cache_ttl}s)"
                )
            except Exception as exc:
                logger.warning(
                    f"⚠️ Financial insight cache write failed for user {user_id}: {exc}"
                )

            return {
                "status": "success",
                "user_id": user_id,
                "month": ym,
                "insights": insights,
                "has_data": has_data_flag,
                "error": None,
            }
        except Exception as exc:
            logger.error(f"❌ Financial insight error for user {user_id}: {exc}")
            return {
                "status": "error",
                "user_id": user_id,
                "month": ym,
                "insights": [],
                "has_data": False,
                "error": str(exc),
            }

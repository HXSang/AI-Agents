from __future__ import annotations

import asyncio
import hashlib
import json
import re
from calendar import month_name, monthrange
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np

from agents.llm_helper import llm_response_text
from agents.llm_helper import resolve_generation_language
from agents.llm_helper import translate_insight_dict

from agents.llm_manager import LLMManager
from agents.prompt import get_language_name 
from agents.prompt import important_language_prompt
from clients.redis_client import RedisClient
from insights.services.monthly_base import MonthlyInsightService
from langchain_core.messages import SystemMessage
from langchain_core.messages import HumanMessage

from models.models import DailySnapshot
from models.models import Insight
from models.models import InsightCandidate
from models.models import MonthlyInsightResponse
from models.models import MonthlySnapshot
from models.models import PhaseMetrics

from services.executor.cache_helpers import CacheHelpers
from services.external_api_service import IExternalAPIService
from utils.logger import logger

Number = Union[int, float]


# =============================================================================
# Baseline context — cross-month historical data per Rules §1
# =============================================================================


@dataclass
class BaselineContext:

    prev_month_snapshot: Optional[MonthlySnapshot] = None
    prev_month_insights: List[Insight] = field(default_factory=list)
    avg_90d: Dict[str, Optional[float]] = field(default_factory=dict)
    avg_180d: Dict[str, Optional[float]] = field(default_factory=dict)
    past_insights_by_month: Dict[Tuple[int, int], List[Insight]] = field(
        default_factory=dict
    )

    def has_prev_month(self) -> bool:
        return self.prev_month_snapshot is not None

    def has_90d_baseline(self) -> bool:
        return bool(self.avg_90d)

    def has_180d_baseline(self) -> bool:
        return bool(self.avg_180d)

_BASELINE_METRIC_KEYS: Tuple[str, ...] = (
    "overall_day_score_avg",
    "h_health_score_avg",
    "p_task_completion_rate_avg",
    "m_mood_score_avg",
    "financial_health_score_avg",
)


# =============================================================================
# Constants
# =============================================================================

# Pipeline / orchestration
_TOP_N: int = 3
# Rules §5 row 4 — conflict priority "correlation > trend > level"
_CONFLICT_PRIORITY: Dict[str, int] = {
    "correlation": 3,
    "phase_deterioration": 2,
    "phase_rise": 2,
    "trend": 1,
    "level": 0,
}

# Rollup
_MIN_MODULES_FOR_BALANCE: int = 2

# Phase split
_PHASE_BOUNDS: List[Tuple[int, int, int]] = [(1, 1, 10), (2, 11, 20), (3, 21, 31)]
LATE_WORK_SPAN_MIN: float = 600.0

# Edge-case filter (Rules §5)
_EXTREME_STRENGTH: float = 0.60
_LOW_VOLUME_DAYS: int = 14
_VARIANCE_KEYS: Tuple[str, ...] = (
    "overall_day_score_std",
    "max_phase_sleep_std",
    "phase_score_spread_pct",
    "phase3_overall_day_score_std",
    "phase1_std",
    "phase2_std",
    "growth",
    "variance_growth",
)
_LOW_ACTIVITY_OVERALL_FLOOR: float = 55.0

# Scoring weights (Rules §5)
W_SIGNAL: float = 0.40
W_NOVELTY: float = 0.30
W_CROSS_MODULE: float = 0.30
ANTI_REP_LOOKBACK_MONTHS: int = 3
_STABLE_RHYTHM_P2_MAX_STD: float = 10.0
_SUSTAINED_BALANCE_MIN_AVG: float = 75.0
_PHASE_BALANCE_DROP_MIN: float = 20.0
_EARLY_CALENDAR_DAY_LIMIT: int = 5
_MEETING_HEAVY_DAY_THRESHOLD_MIN: float = 360.0
_MEETING_HEAVY_MIN_DAYS: int = 3
_RECOVERY_REBOUND_BASELINE_DROP_PCT: float = -0.20
_FINAL_DAY_MIN_DAYS: int = 25
_FINAL_DAY_TRADEOFF_CORR: float = 0.50  # |corr| ≥ this → trade-off summary fires
_FINAL_DAY_STABILITY_MAX_SPREAD: float = 0.10  # phase score spread ≤ 10% → stability
_FINAL_DAY_IMPROVEMENT_MIN_GAP: float = 0.20  # ≥20% gap vs baseline → improvement area
_FINAL_DAY_DOMINANT_MIN_EXCESS: float = 0.20  # module excess ≥ 20% → dominant pattern
_FINAL_DAY_HISTCOMP_NA_DROP: float = 0.35

_CARRY_OVER_EXEMPT_TYPES: frozenset = frozenset(
    {
        "positive_carry_over",  # P1.9 — Great Job (same strong pattern as last month)
        "negative_carry_over",  # P1.10 — Need Attention low-confidence
        "habit_alignment_180d",  # P1.11 — Great Job (aligns with 180d trend)
        "pattern_repeat_90d",  # P2.12 — appears in ≥2 of last 3 months
        "reinforcement_90d",  # P3.12 — same outcome across 3 months
        "habit_180d",  # P3.13 — matches long-term trend
        "dominant_pattern_final_day",  # FD.1 — strongest signal
        "tradeoff_summary_final_day",  # FD.2 — most significant trade-off
        "stability_summary_final_day",  # FD.3 — steady across month
        "improvement_area_final_day",  # FD.4 — weakest module
        "historical_comparison_final_day",  # FD.5 — month vs 90d/180d
    }
)

_WEEK_FOCUS_FAMILY: Dict[int, str] = {
    1: "B",  # Week 1 (days 1-7)   → Time patterns
    2: "X",  # Week 2 (days 8-14)  → Cross-module
    3: "C",  # Week 3 (days 15-21) → Behaviour sequences
    4: "D",  # Week 4 (days 22+)   → Goal alignment
}
_WEEK_FOCUS_BOOST: float = 0.10

# Family A — Calendar
_FOCUS_BLOCKS_PER_DAY: float = 2.5
_TASK_COMPLETION_GOOD: float = 75.0
_HIGH_EVENT_DENSITY: float = 5.0
_SHORT_EVENT_MIN_CEIL: float = 30.0
_FEWER_MEETINGS_CEILING: float = 180.0
_SOCIAL_ENERGY_CORR: float = 0.30
_SOCIAL_ENERGY_MEETING_FLOOR: float = 120.0
_SOCIAL_ENERGY_MOOD_FLOOR: float = 3.0

_LATE_EVENING_DAYS_FRACTION: float = 0.30
_SLEEP_HOURS_STABLE_STD: float = 0.50
_SLEEP_PHASE_DRIFT_PCT: float = 0.10
_SLEEP_HOURS_NOISY_STD: float = 1.00
_OVERALL_NOISY_STD: float = 10.0

# Family C — Sequence
_NEG_CORR_THRESHOLD: float = -0.5
_POS_CORR_THRESHOLD: float = 0.5
_SLEEP_DROP_PCT: float = -0.15
_WORK_RISE_PCT: float = 0.20
_ACTIVITY_DROP_PCT: float = -0.20

_BUSY_SPEND_CORR: float = 0.30
_BUSY_SPEND_CORR_CEILING: float = 0.40
_BUSY_EVENTS_PER_DAY: float = 3.0

# Family D — Goal vs Reality
_ALIGNED_FLOOR: float = 0.80
_MISALIGNED_CEILING: float = 0.50

# Family F — Finance context
_CORR_STRESS_SPEND: float = 0.50
_CORR_EVENT_SPEND: float = 0.40
_CORR_CONTROLLED_CEILING: float = 0.20
_SPEND_RISE_PCT: float = 0.30
_WORK_RISE_PCT_FOR_SPEND: float = 0.20
_CONTROLLED_MEETING_MIN: float = 180.0

# Family X — Cross-phase
_STABLE_PCT: float = 0.10
_TREND_DECLINE: float = -0.20
_TREND_BUILD: float = 0.15
_LATE_CONCENTRATION_PCT: float = 0.20
_SKEW_EXCESS_PCT: float = 0.30
_EARLY_LOW_SCORE: float = 50.0
_EARLY_DAYS_CAP: int = 10
_BALANCED_START_MAX_EXCESS: float = 0.15

_IMBALANCE_HIGH_STD: float = 15.0
_IRREGULAR_FINISH_STD: float = 12.0  # Phase 3 overall_day_score_std
_CATCHUP_PCT: float = 0.25  # P3 vs P2 score uplift after a P2 dip
# Rules §1 Phase 3 row 3 — "Strong finish: Phase 3 > Phase 2 significantly" (no dip needed)
_STRONG_FINISH_PCT: float = 0.15
_FINANCE_LATE_DOM_PCT: float = 0.20  # P3 financial vs avg(P1,P2)
_RECOVERY_REBOUND_PCT: float = 0.15  # P2 health vs P1 health

_EARLY_VARIABILITY_STD: float = 12.0  # Phase 1 overall_day_score_std → Opportunity
_INCREASING_VARIANCE_PCT: float = 0.30  # P2 std vs P1 std → Need Attention
_CROSS_STABLE_STD_CEIL: float = 8.0  # std ceiling for cross-module stability
_CROSS_STABLE_CORR_CEIL: float = 0.30  # |corr| ceiling for stability claim
_CONTROLLED_INCREASE_PCT: float = 0.20  # P3 work uplift while others ±10% stable
_CONTROLLED_INCREASE_STABLE_PCT: float = 0.10
_STRESS_PATTERN_WORK_UP: float = 0.20  # P3 work uplift vs P2
_STRESS_PATTERN_SLEEP_DOWN: float = -0.10
_STRESS_PATTERN_VAR_GROW: float = 0.20  # P3 std vs P2 std

_LEVEL_VS_HISTORY_PCT: float = 0.20  # ≥20% above/below baseline → high/low start
_START_CHANGE_PCT: float = 0.20  # ≥20% different from last month's start

_CARRY_OVER_MIN_SIGNAL: float = 0.40  # last-month insight must have score ≥ this

_POSITIVE_CARRY_OVER_TYPES: frozenset = frozenset(
    {
        "stable_rhythm",  # §1 P3 row 2 — consistent across phases
        "consistent_routine",  # §3.B row 2 — stable sleep schedule
        "balanced_start",  # §1 P1 row 5 — phase-1 balance
        "sustained_balance",  # §1 P2 row 5 — P1+P2 balance ≥ 75%
        "cross_module_stability",  # §1 P2 row 10 — low variance + low corr
        "focused_work_blocks",  # §3.A row 2 — durable focus pattern
        "social_energy_days",  # §3.A row 4 — durable mood-meeting link
        "controlled_spending",  # §3.F row 2 — durable expense control
        "workout_mood_chain",  # §3.C row 2 — durable workout-mood link
        "consistent_goal_alignment",  # §3.D row 1 — durable goal alignment
        "habit_alignment_180d",  # §1 P1 row 11 — 180d pattern match
        "habit_180d",  # §1 P3 row 13 — 180d pattern match
    }
)

_NEGATIVE_CARRY_OVER_TYPES: frozenset = frozenset(
    {
        "cross_module_imbalance",  # §1 P3 row 8 — high balance score
        "skewed_allocation",  # §1 P1 row 4 — module excess ≥ 30%
        "growing_imbalance",  # §1 P2 row 6 — balance dropping
        "deterioration_vs_last_month",  # §1 P3 row 11 — worse vs prev
        "misalignment",  # §3.D row 3 — goal gap across modules
        "work_health_tradeoff_corr",  # §1 P2 row 7 — work↑+health↓ corr
    }
)

_HABIT_180D_MATCH_PCT: float = 0.10  # current within ±10% of 180d avg = aligned

_DIVERGENCE_VS_HISTORY_PCT: float = 0.20  # ≥20% different from prev month mid

_PATTERN_REPEAT_MIN_OCCURRENCES: int = 2  # ≥2 of last 3 months show same pattern

_BEHAVIOUR_SHIFT_PCT: float = 0.20  # ≥20% deviation from 180d baseline

_MONTH_CHANGE_BAL_PCT: float = 10.0  # balance change vs prev month threshold

_REINFORCEMENT_PAST_MIN: int = 2

_BASELINE_LOOKBACK_MONTHS: int = 6

_MOOD_VOLATILITY_RANGE: float = 1.5  # daily mood (max-min) average ≥ 1.5 (on 1-5 scale)


# =============================================================================
# Statistical helpers
# =============================================================================


def _clean(values: Iterable[Any]) -> List[Any]:
    return [v for v in values if v is not None]


def safe_mean(values: Iterable[Optional[Number]]) -> Optional[float]:
    vals = _clean(values)
    if not vals:
        return None
    return round(sum(vals) / len(vals), 2)


def safe_std(values: Iterable[Optional[Number]]) -> Optional[float]:
    """Sample std (ddof=1). Returns None if N < 2."""
    vals = _clean(values)
    if len(vals) < 2:
        return None
    return round(float(np.std(vals, ddof=1)), 2)


def safe_sum(values: Iterable[Optional[Number]]) -> Optional[Number]:
    vals = _clean(values)
    if not vals:
        return None
    s = sum(vals)
    if all(isinstance(v, int) and not isinstance(v, bool) for v in vals):
        return int(s)
    return round(s, 2)


def safe_pearson(
    x: Sequence[Optional[Number]],
    y: Sequence[Optional[Number]],
) -> Optional[float]:

    pairs: List[Tuple[float, float]] = [
        (float(a), float(b)) for a, b in zip(x, y) if a is not None and b is not None
    ]
    if len(pairs) < 3:
        return None
    a_arr = np.array([p[0] for p in pairs], dtype=float)
    b_arr = np.array([p[1] for p in pairs], dtype=float)
    if a_arr.std() == 0 or b_arr.std() == 0:
        return None
    r = float(np.corrcoef(a_arr, b_arr)[0, 1])
    if np.isnan(r):
        return None
    return round(r, 3)


def pct_delta(new: Optional[Number], old: Optional[Number]) -> Optional[float]:
    if new is None or old is None or old == 0:
        return None
    return round((new - old) / abs(old), 4)


# =============================================================================
# Phase split + monthly rollup
# =============================================================================


def split_phases(
    dailies: List[DailySnapshot], year: int, month: int
) -> List[PhaseMetrics]:
    """Split month into 3 phases. Always returns exactly 3 PhaseMetrics."""
    last_day = monthrange(year, month)[1]
    by_day = {d.snapshot_date: d for d in dailies}

    out: List[PhaseMetrics] = []
    for phase_no, lo, hi in _PHASE_BOUNDS:
        hi_clamped = min(hi, last_day)
        phase_start = date(year, month, lo)
        phase_end = date(year, month, hi_clamped)
        days = [
            by_day[date(year, month, d)]
            for d in range(lo, hi_clamped + 1)
            if date(year, month, d) in by_day
        ]
        out.append(_aggregate_phase(phase_no, phase_start, phase_end, days))
    return out


def _aggregate_phase(
    phase: int,
    start: date,
    end: date,
    days: List[DailySnapshot],
) -> PhaseMetrics:
    if not days:
        return PhaseMetrics(
            phase=phase,  # type: ignore[arg-type]
            phase_start_date=start,
            phase_end_date=end,
            days_with_data=0,
        )

    return PhaseMetrics(
        phase=phase,  # type: ignore[arg-type]
        phase_start_date=start,
        phase_end_date=end,
        days_with_data=len(days),
        # Health
        h_steps_avg=safe_mean(d.h_steps for d in days),
        h_sleep_hours_avg=safe_mean(d.h_sleep_hours for d in days),
        h_sleep_hours_std=safe_std(d.h_sleep_hours for d in days),
        h_sleep_quality_avg=safe_mean(d.h_sleep_quality for d in days),
        h_health_score_avg=safe_mean(d.h_health_score for d in days),
        h_active_minutes_avg=safe_mean(d.h_active_minutes for d in days),
        h_total_workout_min_sum=safe_sum(d.h_total_workout_min for d in days),
        # Productivity
        p_meeting_minutes_avg=safe_mean(d.p_meeting_minutes for d in days),
        p_total_events_sum=safe_sum(d.p_total_events for d in days),
        p_focus_blocks_30min_sum=safe_sum(d.p_focus_blocks_30min for d in days),
        p_focus_blocks_60min_sum=safe_sum(d.p_focus_blocks_60min for d in days),
        p_task_completion_rate_avg=safe_mean(d.p_task_completion_rate for d in days),
        p_late_evening_event_days=sum(
            1
            for d in days
            if d.p_work_span_minutes is not None
            and d.p_work_span_minutes > LATE_WORK_SPAN_MIN
        ),
        p_heavy_meeting_days=sum(
            1
            for d in days
            if d.p_meeting_minutes is not None
            and d.p_meeting_minutes > _MEETING_HEAVY_DAY_THRESHOLD_MIN
        ),
        p_reminders_due_sum=safe_sum(d.p_reminders_due for d in days),
        p_reminders_completed_sum=safe_sum(d.p_reminders_completed for d in days),
        p_reminders_overdue_sum=safe_sum(d.p_reminders_overdue for d in days),
        # Mood
        m_mood_score_avg=safe_mean(d.m_mood_score_avg for d in days),
        m_mood_score_std=safe_std(d.m_mood_score_avg for d in days),
        m_mood_range_avg=safe_mean(
            (d.m_mood_score_max - d.m_mood_score_min)
            for d in days
            if d.m_mood_score_max is not None and d.m_mood_score_min is not None
        ),
        # Finance
        f_total_expense_sum=safe_sum(d.f_total_expense for d in days),
        f_transaction_count_sum=safe_sum(d.f_transaction_count for d in days),
        f_bills_overdue_sum=safe_sum(d.f_bills_overdue for d in days),
        f_bills_due_today_amount_sum=safe_sum(d.f_bills_due_today_amount for d in days),
        f_budgets_over_limit_max=(
            max(
                (
                    d.f_budgets_over_limit
                    for d in days
                    if d.f_budgets_over_limit is not None
                ),
                default=None,
            )
        ),
        f_budgets_critical_max=(
            max(
                (
                    d.f_budgets_critical
                    for d in days
                    if d.f_budgets_critical is not None
                ),
                default=None,
            )
        ),
        # Composite
        wellness_score_avg=safe_mean(d.wellness_score for d in days),
        productivity_score_avg=safe_mean(d.productivity_score for d in days),
        financial_health_score_avg=safe_mean(d.financial_health_score for d in days),
        overall_day_score_avg=safe_mean(d.overall_day_score for d in days),
        overall_day_score_std=safe_std(d.overall_day_score for d in days),
    )


def compute_monthly(
    dailies: List[DailySnapshot],
    profile_id: str,
    year: int,
    month: int,
    phases: List[PhaseMetrics],
) -> MonthlySnapshot:
    if not 1 <= month <= 12:
        raise ValueError(f"month must be 1..12, got {month}")
    if len(phases) != 3:
        raise ValueError(f"phases must have 3 items, got {len(phases)}")

    last_day = monthrange(year, month)[1]
    month_start = date(year, month, 1)
    month_end = date(year, month, last_day)
    days = [d for d in dailies if month_start <= d.snapshot_date <= month_end]

    corr_work_sleep = safe_pearson(
        [d.p_meeting_minutes for d in days],
        [d.h_sleep_hours for d in days],
    )
    corr_work_health = safe_pearson(
        [d.p_meeting_minutes for d in days],
        [d.h_health_score for d in days],
    )
    corr_work_spend = safe_pearson(
        [d.p_meeting_minutes for d in days],
        [d.f_total_expense for d in days],
    )
    corr_sleep_mood = safe_pearson(
        [d.h_sleep_hours for d in days],
        [d.m_mood_score_avg for d in days],
    )
    corr_workout_mood = safe_pearson(
        [d.h_total_workout_min for d in days],
        [d.m_mood_score_avg for d in days],
    )
    corr_meeting_mood = safe_pearson(
        [d.p_meeting_minutes for d in days],
        [d.m_mood_score_avg for d in days],
    )
    # Rules §3.C row 2 "Workout → mood → productivity" — productivity leg uses
    # daily workout vs productivity correlation so the chain is verified end-
    # to-end (mood↔productivity floor would not).
    corr_workout_productivity = safe_pearson(
        [d.h_total_workout_min for d in days],
        [d.productivity_score for d in days],
    )

    balance_score, concentrated, module_scores, module_excess = _module_balance(days)

    expense_by_cat: Dict[str, float] = {}
    for d in days:
        if not d.f_expense_by_category:
            continue
        for cat, amt in d.f_expense_by_category.items():
            if amt is None:
                continue
            try:
                expense_by_cat[cat] = expense_by_cat.get(cat, 0.0) + float(amt)
            except (TypeError, ValueError):
                continue
    expense_by_cat_rounded = (
        {k: round(v, 2) for k, v in expense_by_cat.items()} if expense_by_cat else None
    )

    return MonthlySnapshot(
        profile_id=profile_id,
        year=year,
        month=month,
        month_start_date=month_start,
        month_end_date=month_end,
        days_with_data=len(days),
        phases=phases,
        h_sleep_hours_avg=safe_mean(d.h_sleep_hours for d in days),
        h_health_score_avg=safe_mean(d.h_health_score for d in days),
        h_total_workout_min_sum=safe_sum(d.h_total_workout_min for d in days),
        p_meeting_minutes_avg=safe_mean(d.p_meeting_minutes for d in days),
        p_task_completion_rate_avg=safe_mean(d.p_task_completion_rate for d in days),
        p_total_events_sum=safe_sum(d.p_total_events for d in days),
        productivity_score_avg=safe_mean(d.productivity_score for d in days),
        p_reminders_due_sum=safe_sum(d.p_reminders_due for d in days),
        p_reminders_completed_sum=safe_sum(d.p_reminders_completed for d in days),
        p_reminders_overdue_sum=safe_sum(d.p_reminders_overdue for d in days),
        m_mood_score_avg=safe_mean(d.m_mood_score_avg for d in days),
        m_mood_range_avg=safe_mean(
            (d.m_mood_score_max - d.m_mood_score_min)
            for d in days
            if d.m_mood_score_max is not None and d.m_mood_score_min is not None
        ),
        f_total_expense_sum=safe_sum(d.f_total_expense for d in days),
        financial_health_score_avg=safe_mean(d.financial_health_score for d in days),
        f_bills_overdue_total=safe_sum(d.f_bills_overdue for d in days),
        f_bills_due_today_amount_total=safe_sum(
            d.f_bills_due_today_amount for d in days
        ),
        f_budgets_over_limit_max=(
            max(
                (
                    d.f_budgets_over_limit
                    for d in days
                    if d.f_budgets_over_limit is not None
                ),
                default=None,
            )
        ),
        f_expense_by_category_sum=expense_by_cat_rounded,
        overall_day_score_avg=safe_mean(d.overall_day_score for d in days),
        overall_day_score_std=safe_std(d.overall_day_score for d in days),
        corr_work_sleep=corr_work_sleep,
        corr_work_health=corr_work_health,
        corr_work_spend=corr_work_spend,
        corr_sleep_mood=corr_sleep_mood,
        corr_workout_mood=corr_workout_mood,
        corr_meeting_mood=corr_meeting_mood,
        corr_workout_productivity=corr_workout_productivity,
        balance_score=balance_score,
        most_concentrated_area=concentrated,
        module_scores=module_scores,
        module_max_excess_pct=module_excess,
        goal_alignment_health_pct=safe_mean(d.h_steps_goal_pct for d in days),
        goal_alignment_productivity_pct=safe_mean(
            d.p_task_completion_rate for d in days
        ),
        goal_alignment_finance_pct=safe_mean(d.f_goals_progress_avg for d in days),
    )


def _module_balance(
    days: List[DailySnapshot],
) -> Tuple[Optional[float], Optional[str], Optional[Dict[str, float]], Optional[float]]:

    h = safe_mean(d.h_health_score for d in days)
    p = safe_mean(d.productivity_score for d in days)
    f = safe_mean(d.financial_health_score for d in days)

    modules: dict = {}
    if h is not None:
        modules["health"] = float(h)
    if p is not None:
        modules["productivity"] = float(p)
    if f is not None:
        modules["finance"] = float(f)

    if len(modules) < _MIN_MODULES_FOR_BALANCE:
        return None, None, None, None

    balance = round(sum(modules.values()) / len(modules), 2)
    concentrated = max(modules, key=modules.get)

    max_score = modules[concentrated]
    others = [v for k, v in modules.items() if k != concentrated]
    others_mean = sum(others) / len(others) if others else None
    if others_mean is None or others_mean <= 0:
        max_excess = None
    else:
        max_excess = round((max_score - others_mean) / others_mean, 3)

    rounded_modules = {k: round(v, 2) for k, v in modules.items()}
    return balance, concentrated, rounded_modules, max_excess


def _phase_balance(phase: PhaseMetrics) -> Optional[float]:
    """Per-phase balance score = arithmetic mean of 3 module composites.

    Mirrors ``_module_balance`` at phase level — per spec, balance_score =
    mean(h_health_score_avg, productivity_score_avg, financial_health_score_avg).
    Mood is excluded (no balance pattern in Rules references mood).

    Per Rules §5 row 2: missing module data → exclude from balance. Requires
    ≥ ``_MIN_MODULES_FOR_BALANCE`` of the 3 modules to have data this phase.

    Returns scale 0..100 (NOT normalized). HIGH = doing well, LOW = under.
    """
    modules: List[float] = []
    if phase.h_health_score_avg is not None:
        modules.append(float(phase.h_health_score_avg))
    if phase.productivity_score_avg is not None:
        modules.append(float(phase.productivity_score_avg))
    if phase.financial_health_score_avg is not None:
        modules.append(float(phase.financial_health_score_avg))
    if len(modules) < _MIN_MODULES_FOR_BALANCE:
        return None
    return round(sum(modules) / len(modules), 2)


def _monthly_imbalance_std(snapshot: MonthlySnapshot) -> Optional[float]:
    """Month-level imbalance std (3 modules, no mood)."""
    modules: List[float] = []
    if snapshot.h_health_score_avg is not None:
        modules.append(float(snapshot.h_health_score_avg))
    if snapshot.productivity_score_avg is not None:
        modules.append(float(snapshot.productivity_score_avg))
    if snapshot.financial_health_score_avg is not None:
        modules.append(float(snapshot.financial_health_score_avg))
    if len(modules) < _MIN_MODULES_FOR_BALANCE:
        return None
    return round(float(safe_std(modules) or 0.0), 2)


# =============================================================================
# Anti-rep + scoring
# =============================================================================


def insight_hash(candidate: InsightCandidate) -> str:
    """Stable 16-char hex hash of (family, type, phase_focus). Signal not hashed."""
    key = f"{candidate.pattern_family}:{candidate.pattern_type}:{candidate.phase_focus}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]

_RECENCY_PENALTY_BY_AGE: Dict[int, float] = {
    1: 1.00, 
    2: 0.50,
    3: 0.25,
}
_RECENCY_PENALTY_SCALE: float = 0.5


def novelty_score(
    candidate: InsightCandidate,
    recent_insights: Optional[List[Insight]] = None,
    past_insights_by_month: Optional[Dict[Tuple[int, int], List[Insight]]] = None,
    current_year: Optional[int] = None,
    current_month: Optional[int] = None,
) -> float:
    if candidate.pattern_type in _CARRY_OVER_EXEMPT_TYPES:
        return 1.0

    h = insight_hash(candidate)

    # Preferred path — recency weighted (needs month-keyed dict + anchor month)
    if (
        past_insights_by_month
        and current_year is not None
        and current_month is not None
    ):
        total_penalty = 0.0
        for (y, m), insights in past_insights_by_month.items():
            age = (current_year - y) * 12 + (current_month - m)
            if age <= 0:
                continue
            weight = _RECENCY_PENALTY_BY_AGE.get(age, 0.0)
            if weight == 0.0:
                continue
            matches = sum(1 for i in insights if i.insight_hash == h)
            if matches:
                total_penalty += weight * matches
        score = 1.0 - _RECENCY_PENALTY_SCALE * total_penalty
        return max(0.0, round(score, 3))

    # Legacy path — flat list with no recency info
    if not recent_insights:
        return 1.0
    matches = sum(1 for i in recent_insights if i.insight_hash == h)
    if matches == 0:
        return 1.0
    if matches == 1:
        return 0.5
    return 0.0


def compute_score(candidate: InsightCandidate, novelty: float) -> float:
    return round(
        candidate.signal_strength * W_SIGNAL
        + novelty * W_NOVELTY
        + candidate.cross_module_impact * W_CROSS_MODULE,
        3,
    )


def _infer_trigger(candidate: InsightCandidate) -> str:
    
    explicit = candidate.raw_signal.get("trigger")
    if explicit:
        return explicit
    keys = candidate.raw_signal.keys()
    if any(k.startswith("corr_") for k in keys):
        return "correlation"
    if any(("_delta" in k) or ("_vs_" in k) for k in keys):
        return "trend"
    return "level"


def _conflict_priority(candidate: InsightCandidate) -> int:
    return _CONFLICT_PRIORITY.get(_infer_trigger(candidate), 0)

_PRIORITY_SCORE_BOOST: Dict[str, float] = {
    "correlation": 0.05,
    "phase_deterioration": 0.03,
    "phase_rise": 0.03,
    "trend": 0.02,
    "level": 0.00,
}


def _apply_conflict_priority_boost(
    scored: List[Tuple[InsightCandidate, float, float]],
) -> List[Tuple[InsightCandidate, float, float]]:

    if not scored:
        return scored
    out: List[Tuple[InsightCandidate, float, float]] = []
    for cand, score, novelty in scored:
        trigger = _infer_trigger(cand)
        boost = _PRIORITY_SCORE_BOOST.get(trigger, 0.0)
        new_score = round(score + boost, 3) if boost else score
        out.append((cand, new_score, novelty))
    return out

_FINAL_DAY_PATTERN_TYPES: frozenset = frozenset(
    {
        "dominant_pattern_final_day",
        "tradeoff_summary_final_day",
        "stability_summary_final_day",
        "improvement_area_final_day",
        "historical_comparison_final_day",
    }
)

_FINAL_DAY_WINDOW_MAX_DOM: int = 2

_PHASE1_WINDOW_MAX_DOM: int = 10
_PHASE2_WINDOW_MAX_DOM: int = 20

# ``phase_focus`` values that bind a pattern to ONE specific phase. Patterns
# outside this set (``"all"`` whole-month, ``"cross"`` cross-phase — incl. every
# Family A-F pattern) are never phase-gated and always run.
_PHASE_SPECIFIC_FOCUSES: frozenset = frozenset({"1", "2", "3"})


def _now_in_tz(timezone: Optional[str]) -> datetime:
    """``now`` in the user's timezone; falls back to UTC if missing / invalid.

    Single source of truth for "what day is it for the user" shared by the
    month-resolution and final-day-window helpers below.
    """
    tz = None
    if timezone:
        try:
            tz = ZoneInfo(timezone)
        except (ZoneInfoNotFoundError, ValueError):
            logger.warning(f"⚠️ Unknown timezone {timezone!r}, falling back to UTC")
    return datetime.now(tz) if tz else datetime.utcnow()


def _current_year_month(timezone: Optional[str]) -> Tuple[int, int]:
    """Resolve "now" in the user's timezone → (year, month)."""
    now = _now_in_tz(timezone)
    return now.year, now.month


def _shift_if_final_day_window(
    year: int, month: int, timezone: Optional[str]
) -> Tuple[int, int]:
    """On days 1-2 (user tz), redirect a CURRENT-month target to the PREVIOUS
    month so the month-end "final day" synthesis has a complete month of data
    (spec: shown on the 1st-2nd of the following month). Applies to BOTH explicit
    and auto-derived requests; a request for any OTHER specific month is left
    untouched.
    """
    now = _now_in_tz(timezone)
    if now.day <= _FINAL_DAY_WINDOW_MAX_DOM and (year, month) == (now.year, now.month):
        prev_year, prev_month = now.year, now.month - 1
        if prev_month == 0:
            prev_month = 12
            prev_year -= 1
        logger.info(
            f"🗓️ Day {now.day} → redirecting current-month target "
            f"{year}-{month:02d} to previous month {prev_year}-{prev_month:02d}"
        )
        return prev_year, prev_month
    return year, month


def _is_final_day_window(year: int, month: int, timezone: Optional[str]) -> bool:
    """True when "now" in the user's tz is day 1-2 of the month right AFTER the
    analysed ``(year, month)`` — i.e. we're rendering last month's wrap-up.
    """
    now = _now_in_tz(timezone)
    if now.day > _FINAL_DAY_WINDOW_MAX_DOM:
        return False
    prev_year, prev_month = now.year, now.month - 1
    if prev_month == 0:
        prev_month = 12
        prev_year -= 1
    return (year, month) == (prev_year, prev_month)


def _active_phase_focus(year: int, month: int, timezone: Optional[str]) -> str:
    """Return the phase ("1"/"2"/"3") whose patterns are surfaced right now.

    - Analysed month == current calendar month (user tz): derive from today's
      day-of-month — days 3-10 → Phase 1, 11-20 → Phase 2, 21-EOM → Phase 3.
      (Days 1-2 are the final-day window, handled by ``_is_final_day_window``;
      they never reach this phase-window path.)
    - Any OTHER (already-complete) month: Phase 3 — the month is done, so the
      end-of-month view is the honest one.
    """
    now = _now_in_tz(timezone)
    if (year, month) != (now.year, now.month):
        return "3"
    if now.day <= _PHASE1_WINDOW_MAX_DOM:
        return "1"
    if now.day <= _PHASE2_WINDOW_MAX_DOM:
        return "2"
    return "3"


def _apply_phase_window(
    candidates: List[InsightCandidate], active_phase: str
) -> List[InsightCandidate]:
    """Time-gated detection — drop phase-specific patterns of OTHER phases.

    Keeps:
      - ``phase_focus == active_phase`` (this phase's own patterns)
      - ``phase_focus`` in {"all", "cross"} — whole-month / cross-module
        patterns, incl. every Family A-F pattern (never phase-gated).

    So during the Phase-3 calendar window a Phase-1 ``skewed_allocation`` no
    longer leaks in, while ``meeting_overload`` (A) / ``work_health_tradeoff_corr``
    (cross) still run. Final-day patterns are gated upstream and never reach here.
    """
    return [
        c
        for c in candidates
        if c.phase_focus not in _PHASE_SPECIFIC_FOCUSES or c.phase_focus == active_phase
    ]


def _dedupe(
    scored: List[Tuple[InsightCandidate, float, float]],
) -> List[Tuple[InsightCandidate, float, float]]:
    """Keep one candidate per insight_hash. Tie-break: priority, then score."""
    by_hash: Dict[str, Tuple[InsightCandidate, float, float]] = {}
    for triple in scored:
        c, score, _ = triple
        h = insight_hash(c)
        if h not in by_hash:
            by_hash[h] = triple
            continue
        existing_c, existing_score, _ = by_hash[h]
        cur_pri = _conflict_priority(c)
        ex_pri = _conflict_priority(existing_c)
        if cur_pri > ex_pri or (cur_pri == ex_pri and score > existing_score):
            by_hash[h] = triple
    return list(by_hash.values())


def _week_of_month_today(
    year: int,
    month: int,
    today: Optional[date] = None,
) -> Optional[int]:
    """Return week-of-month bucket (1..4) for today IF the analysed month is
    the current calendar month, else None (no week boost for past months).

    Buckets: days 1-7 → 1, 8-14 → 2, 15-21 → 3, 22+ → 4.
    """
    t = today or date.today()
    if t.year != year or t.month != month:
        return None
    day = t.day
    if day <= 7:
        return 1
    if day <= 14:
        return 2
    if day <= 21:
        return 3
    return 4


def _apply_week_focus_boost(
    scored: List[Tuple[InsightCandidate, float, float]],
    year: int,
    month: int,
    today: Optional[date] = None,
) -> List[Tuple[InsightCandidate, float, float]]:
    """Rules §4 Rule 1: Rotate insight types by week.

    Applies ``_WEEK_FOCUS_BOOST`` to score of candidates whose ``pattern_family``
    matches this week's focus family. Past months (or anytime ``today`` falls
    outside ``year``/``month``) get no boost — week rotation is a "live"
    in-month behaviour, not historical.
    """
    bucket = _week_of_month_today(year, month, today)
    if bucket is None:
        return scored
    focus_family = _WEEK_FOCUS_FAMILY.get(bucket)
    if focus_family is None:
        return scored
    boosted: List[Tuple[InsightCandidate, float, float]] = []
    for cand, score, novelty in scored:
        new_score = (
            score + _WEEK_FOCUS_BOOST if cand.pattern_family == focus_family else score
        )
        boosted.append((cand, round(new_score, 3), novelty))
    return boosted


def _pick_one_per_category(
    scored: List[Tuple[InsightCandidate, float, float]],
    n: int = _TOP_N,
) -> List[Tuple[InsightCandidate, float, float]]:
    """Rules §6 Final Output Format: prefer 1 of each category in top-N.

    Strategy:
        1. From each of [Need Attention, Opportunity, Great Job] pick the
           highest-score candidate (skip categories with no candidate).
        2. If fewer than ``n`` were picked (a category had no candidate),
           fill from remaining candidates by score-desc.
        3. Re-sort the picked set by score-desc for stable rank order.

    This guarantees 3-category coverage when data permits, while still
    honouring Rules §5 "top 3 by score" as the tie-breaker / filler.

    Week-based family rotation (Rules §4 Rule 1) is applied UPSTREAM via
    ``_apply_week_focus_boost`` so the highest-score candidate per category
    already reflects the current week's family preference.
    """
    if not scored:
        return []

    by_category: Dict[str, List[Tuple[InsightCandidate, float, float]]] = {
        "Need Attention": [],
        "Opportunity": [],
        "Great Job": [],
    }
    for triple in scored:
        cat = triple[0].tentative_category
        if cat in by_category:
            by_category[cat].append(triple)
    for cat in by_category:
        by_category[cat].sort(key=lambda t: t[1], reverse=True)

    picked: List[Tuple[InsightCandidate, float, float]] = []
    seen_ids: set = set()
    for cat in ("Need Attention", "Opportunity", "Great Job"):
        if by_category[cat]:
            chosen = by_category[cat][0]
            picked.append(chosen)
            seen_ids.add(id(chosen))

    # Fill if we have fewer than n (a category was empty).
    if len(picked) < n:
        remaining = sorted(
            (t for t in scored if id(t) not in seen_ids),
            key=lambda t: t[1],
            reverse=True,
        )
        for triple in remaining:
            if len(picked) >= n:
                break
            picked.append(triple)
            seen_ids.add(id(triple))

    # Stable rank assignment by score-desc.
    picked.sort(key=lambda t: t[1], reverse=True)
    return picked[:n]


# =============================================================================
# Edge-case filter (Rules §5)
# =============================================================================


def apply_filter(
    candidates: List[InsightCandidate], snapshot: MonthlySnapshot
) -> List[InsightCandidate]:
    out = _drop_early_weak(candidates, snapshot)
    out = _downgrade_low_volume(out, snapshot)
    out = _downgrade_low_activity(out, snapshot)
    return out


def _drop_early_weak(
    candidates: List[InsightCandidate], snapshot: MonthlySnapshot
) -> List[InsightCandidate]:
    """Rules §5 row 1: 'Days 1-5 only → suppress Need Attention unless extreme'.

    Spec keys this on CALENDAR day-of-month (the user's date), not data
    sparsity. A mid-month user with only 1 day of recorded data should still
    surface a Need-Attention insight if the signal is real. Low-volume sparse-
    data handling lives in ``_downgrade_low_volume`` instead.
    """
    today = date.today()
    in_early_calendar_window = (
        today.year == snapshot.year
        and today.month == snapshot.month
        and today.day <= _EARLY_CALENDAR_DAY_LIMIT
    )
    if not in_early_calendar_window:
        return candidates
    return [
        c
        for c in candidates
        if not (
            c.tentative_category == "Need Attention"
            and c.signal_strength < _EXTREME_STRENGTH
        )
    ]


def _downgrade_low_volume(
    candidates: List[InsightCandidate], snapshot: MonthlySnapshot
) -> List[InsightCandidate]:
    """Rules §5 row 5: high variance signal + low volume → downgrade to Opportunity."""
    days = snapshot.days_with_data
    if days >= _LOW_VOLUME_DAYS:
        return candidates

    out: List[InsightCandidate] = []
    for c in candidates:
        if c.tentative_category != "Need Attention":
            out.append(c)
            continue
        if c.signal_strength >= _EXTREME_STRENGTH:
            out.append(c)
            continue
        is_variance_signal = any(k in c.raw_signal for k in _VARIANCE_KEYS)
        if not is_variance_signal:
            out.append(c)
            continue
        downgraded = c.model_copy(
            update={
                "tentative_category": "Opportunity",
                "raw_signal": {
                    **c.raw_signal,
                    "downgraded_from": "Need Attention",
                    "downgrade_reason": "Rules §5 row 5: high variance + low volume",
                },
            }
        )
        out.append(downgraded)
    return out


def _downgrade_low_activity(
    candidates: List[InsightCandidate], snapshot: MonthlySnapshot
) -> List[InsightCandidate]:
    """Rules §5 row 3: 'Low activity across all → Classify as Opportunity, not Need Attention'.

    When the month's overall_day_score average is at/under the low-activity
    floor, the data reflects under-engagement (BE defaults to ≈50 when a
    module has no real data). Per spec, Need Attention candidates should
    be reframed as Opportunity, except when the signal is genuinely extreme
    (then NA stays so we don't silence real risk).
    """
    score = snapshot.overall_day_score_avg
    if score is None or score > _LOW_ACTIVITY_OVERALL_FLOOR:
        return candidates

    out: List[InsightCandidate] = []
    for c in candidates:
        if c.tentative_category != "Need Attention":
            out.append(c)
            continue
        if c.signal_strength >= _EXTREME_STRENGTH:
            out.append(c)
            continue
        downgraded = c.model_copy(
            update={
                "tentative_category": "Opportunity",
                "raw_signal": {
                    **c.raw_signal,
                    "downgraded_from": "Need Attention",
                    "downgrade_reason": "Rules §5 row 3: low activity across all",
                },
            }
        )
        out.append(downgraded)
    return out


# =============================================================================
# Family A — Calendar detectors (4)
# =============================================================================


def _detect_family_a(
    snapshot: MonthlySnapshot, phases: List[PhaseMetrics]
) -> List[InsightCandidate]:
    out: List[InsightCandidate] = []
    for fn in (
        _meeting_overload,
        _focused_work_blocks,
        _context_switching,
        _social_energy_days,
    ):
        c = fn(snapshot, phases)
        if c:
            out.append(c)
    return out


def _meeting_overload(
    snapshot: MonthlySnapshot, phases: List[PhaseMetrics]
) -> Optional[InsightCandidate]:
    """Rules §3.A row 1: '>6h meetings/day for multiple days + drop in health metrics' → Need Attention.

    Spec emphasises "for **multiple days**" so we count days where meeting
    minutes exceed ``_MEETING_HEAVY_DAY_THRESHOLD_MIN`` and require at least
    ``_MEETING_HEAVY_MIN_DAYS``. Day-level counting via
    ``PhaseMetrics.p_heavy_meeting_days`` (populated by ``_aggregate_phase``)
    so a single 12-hour day surrounded by zero days no longer triggers — and
    so a phase whose AVERAGE clears 360min but whose distribution is uneven
    isn't over-counted.

    Backwards-compat: when ``p_heavy_meeting_days`` is missing on cached
    snapshots (predates the field), fall back to the phase-average estimate.
    """
    heavy_days = sum((p.p_heavy_meeting_days or 0) for p in phases)
    if heavy_days == 0:
        # Fallback for cached snapshots predating ``p_heavy_meeting_days``.
        phase_heavy = [
            p
            for p in phases
            if p.days_with_data > 0
            and p.p_meeting_minutes_avg is not None
            and p.p_meeting_minutes_avg >= _MEETING_HEAVY_DAY_THRESHOLD_MIN
        ]
        heavy_days = sum(p.days_with_data for p in phase_heavy)
    if heavy_days < _MEETING_HEAVY_MIN_DAYS:
        return None
    avg_mtg = snapshot.p_meeting_minutes_avg or 0.0

    # Health drop signal: negative correlation OR phase3 health 10% lower than phase1
    health_signal = False
    corr = snapshot.corr_work_health
    if corr is not None and corr <= -0.3:
        health_signal = True
    else:
        p1 = phases[0].h_health_score_avg
        p3 = phases[2].h_health_score_avg
        if p1 is not None and p1 > 0 and p3 is not None and (p3 / p1) <= 0.90:
            health_signal = True

    if not health_signal:
        return None

    strength = min(1.0, heavy_days / 14.0 + 0.4)
    return InsightCandidate(
        pattern_family="A",
        pattern_type="meeting_overload",
        tentative_category="Need Attention",
        phase_focus="all",
        raw_template_key="meeting_overload",
        raw_signal={
            "subject": "your meeting load and health",
            "scope": "month_long_meeting_overload",
            "time_window_human": "across the month",
            "improvement_direction": "trimming heavy meeting days to protect recovery",
            "meeting_load_label": _label_meeting_load(avg_mtg),
            "frequency_label": _label_frequency(
                heavy_days / max(snapshot.days_with_data, 1)
            ),
            "health_link": (
                "declining" if (corr is not None and corr <= -0.3) else "phase_drop"
            ),
            "heavy_meeting_days": heavy_days,
            "p_meeting_minutes_avg": avg_mtg,
            "corr_work_health": corr,
        },
        signal_strength=round(strength, 3),
        cross_module_impact=0.50,
    )


def _focused_work_blocks(
    snapshot: MonthlySnapshot, phases: List[PhaseMetrics]
) -> Optional[InsightCandidate]:
    """Rules §3.A row 2: 'Fewer meetings + high task completion' → Great Job."""
    days = snapshot.days_with_data
    if days < 7:
        return None

    avg_mtg = snapshot.p_meeting_minutes_avg
    if avg_mtg is None or avg_mtg > _FEWER_MEETINGS_CEILING:
        return None

    total_blocks = sum((p.p_focus_blocks_30min_sum or 0) for p in phases)
    if total_blocks <= 0:
        return None
    blocks_per_day = total_blocks / days
    if blocks_per_day < _FOCUS_BLOCKS_PER_DAY:
        return None

    completion_avg = safe_mean(p.p_task_completion_rate_avg for p in phases)
    if completion_avg is None or completion_avg < _TASK_COMPLETION_GOOD:
        return None

    strength = min(
        1.0, (blocks_per_day - _FOCUS_BLOCKS_PER_DAY) / _FOCUS_BLOCKS_PER_DAY + 0.4
    )
    return InsightCandidate(
        pattern_family="A",
        pattern_type="focused_work_blocks",
        tentative_category="Great Job",
        phase_focus="all",
        raw_template_key="focused_work_blocks",
        raw_signal={
            "subject": "your focus blocks and task completion",
            "scope": "month_long_focused_work",
            "direction_unambiguous": "stayed high",
            "time_window_human": "across the month",
            "focus_blocks_per_day": round(blocks_per_day, 2),
            "task_completion_avg": completion_avg,
            "p_meeting_minutes_avg": avg_mtg,
        },
        signal_strength=round(strength, 3),
        cross_module_impact=0.25,
    )


def _context_switching(
    snapshot: MonthlySnapshot, phases: List[PhaseMetrics]
) -> Optional[InsightCandidate]:
    """Rules §3.A row 3: 'Many short events (<30 mins) + high variability' → Opportunity.

    Spec explicitly requires "events <30 mins" — we use total minutes / total
    events to derive the average event duration. Falls through if avg duration
    is ≥30 min so a meeting-heavy day with few-but-long meetings isn't flagged.
    """
    days = snapshot.days_with_data
    if days < 7:
        return None

    total_events = snapshot.p_total_events_sum
    if total_events is None or total_events <= 0:
        return None
    events_per_day = total_events / days
    if events_per_day < _HIGH_EVENT_DENSITY:
        return None

    # Average event duration (min) = total meeting minutes / total events
    avg_event_min: Optional[float] = None
    if snapshot.p_meeting_minutes_avg is not None and total_events > 0:
        total_meeting_min = snapshot.p_meeting_minutes_avg * days
        avg_event_min = total_meeting_min / total_events
        if avg_event_min >= _SHORT_EVENT_MIN_CEIL:
            return None

    overall_std = snapshot.overall_day_score_std
    if overall_std is None or overall_std < 8.0:
        return None

    strength = min(
        1.0, (events_per_day / _HIGH_EVENT_DENSITY - 1.0) + (overall_std / 20.0)
    )
    raw_signal: Dict[str, Any] = {
        "subject": "your short events and overall daily activity",
        "scope": "month_long_context_switching",
        "direction_unambiguous": "varied widely",
        "time_window_human": "across the month",
        "improvement_direction": "fewer short events so the daily activity settles",
        "density_label": _label_event_count(events_per_day),
        "event_size_label": (
            "short"
            if (avg_event_min is not None and avg_event_min < _SHORT_EVENT_MIN_CEIL)
            else "mixed"
        ),
        "variability_label": _label_imbalance_std(overall_std),
        "events_per_day": round(events_per_day, 2),
        "overall_day_score_std": overall_std,
    }
    if avg_event_min is not None:
        raw_signal["avg_event_min"] = round(avg_event_min, 1)
    return InsightCandidate(
        pattern_family="A",
        pattern_type="context_switching",
        tentative_category="Opportunity",
        phase_focus="all",
        raw_template_key="context_switching",
        raw_signal=raw_signal,
        signal_strength=round(strength, 3),
        cross_module_impact=0.25,
    )


def _social_energy_days(
    snapshot: MonthlySnapshot, phases: List[PhaseMetrics]
) -> Optional[InsightCandidate]:
    """Rules §3.A row 4: 'Meetings + improved mood' → Great Job.

    Meeting-volume and mood floors remain as sanity gates (insight only
    makes sense when there is enough meeting activity and a baseline-positive
    mood). The correlation, however, is now graded continuously — the
    historical cliff at ``_SOCIAL_ENERGY_CORR`` (0.30) is just an anchor on
    a linear ramp so weaker correlations decay toward the noise floor
    rather than vanish (Feedback §5).
    """
    if snapshot.days_with_data < 7:
        return None

    avg_mtg = snapshot.p_meeting_minutes_avg
    avg_mood = snapshot.m_mood_score_avg
    corr = snapshot.corr_meeting_mood
    if (
        avg_mtg is None
        or avg_mood is None
        or corr is None
        or avg_mtg < _SOCIAL_ENERGY_MEETING_FLOOR
        or avg_mood < _SOCIAL_ENERGY_MOOD_FLOOR
        or corr <= 0
    ):
        return None

    strength = max(0.0, min(1.0, corr))
    if strength < _GRADIENT_NOISE_FLOOR:
        return None
    return InsightCandidate(
        pattern_family="A",
        pattern_type="social_energy_days",
        tentative_category="Great Job",
        phase_focus="all",
        raw_template_key="social_energy_days",
        raw_signal={
            "subject": "meeting-heavy days and your mood",
            "scope": "month_long_meeting_mood_link",
            "direction_unambiguous": "moved together positively",
            "time_window_human": "across the month",
            "p_meeting_minutes_avg": avg_mtg,
            "m_mood_score_avg": avg_mood,
            "corr_meeting_mood": corr,
        },
        signal_strength=round(strength, 3),
        cross_module_impact=0.50,
    )


# =============================================================================
# Family B — Time-of-day detectors (3)
# =============================================================================


def _detect_family_b(
    snapshot: MonthlySnapshot, phases: List[PhaseMetrics]
) -> List[InsightCandidate]:
    out: List[InsightCandidate] = []
    for fn in (_late_work_impact, _consistent_routine, _irregular_timing):
        c = fn(snapshot, phases)
        if c:
            out.append(c)
    return out


def _late_work_impact(
    snapshot: MonthlySnapshot, phases: List[PhaseMetrics]
) -> Optional[InsightCandidate]:
    """Rules §3.B row 1: 'Events after 9pm + sleep ↓' → Need Attention.

    Snapshot lacks ``last_event_time`` so impl uses long work-day span (>10h)
    as proxy. May understate signal vs strict 9pm cutoff.
    """
    days = snapshot.days_with_data
    if days < 7:
        return None

    late_days = sum((p.p_late_evening_event_days or 0) for p in phases)
    frac = late_days / days
    if frac < _LATE_EVENING_DAYS_FRACTION:
        return None

    sleep_avg = snapshot.h_sleep_hours_avg
    if sleep_avg is None or sleep_avg >= 7.0:
        return None

    sleep_deficit = max(0.0, 7.0 - sleep_avg) / 2.0
    strength = min(1.0, (frac - _LATE_EVENING_DAYS_FRACTION) / 0.4 + sleep_deficit)
    return InsightCandidate(
        pattern_family="B",
        pattern_type="late_work_impact",
        tentative_category="Need Attention",
        phase_focus="all",
        raw_template_key="late_work_impact",
        raw_signal={
            "subject": "your late-evening event days and sleep hours",
            "scope": "month_long_late_work_sleep_link",
            "direction_unambiguous": "sleep shortened on days with late evening events",
            "time_window_human": "across the month",
            "improvement_direction": "wrapping events earlier to protect your sleep window",
            "frequency_label": _label_frequency(frac),
            "sleep_label": _label_sleep_hours(sleep_avg),
            "late_evening_day_fraction": round(frac, 3),
            "h_sleep_hours_avg": sleep_avg,
        },
        signal_strength=round(strength, 3),
        cross_module_impact=0.50,
    )


def _consistent_routine(
    snapshot: MonthlySnapshot, phases: List[PhaseMetrics]
) -> Optional[InsightCandidate]:
    """Rules §3.B row 2: 'Stable bedtime/wake' → Great Job.

    Fires whenever at least one phase has enough data to characterise sleep
    stability. When 2-3 phases have data we additionally check inter-phase
    drift; when only one phase has data the drift component is treated as
    zero so early-month users can still surface this Great Job signal.

    Rules §5 row 2: missing module data → exclude. We require sleep means
    to be NON-ZERO — ``sleep_hours == 0`` across the month is BE missing data,
    not a "stable bedtime" pattern. Fail closed when sleep is uniformly 0.
    """
    stds = [p.h_sleep_hours_std for p in phases if p.h_sleep_hours_std is not None]
    if not stds:
        return None
    worst_std = max(stds)
    if worst_std >= _SLEEP_HOURS_STABLE_STD:
        return None

    means = [p.h_sleep_hours_avg for p in phases if p.h_sleep_hours_avg is not None]
    if not means:
        return None
    # 0h across the month = missing tracking, not a stable routine.
    if max(means) <= 0:
        return None
    if len(means) >= 2:
        mean_of_means = sum(means) / len(means)
        if mean_of_means == 0:
            return None
        drift_pct = (max(means) - min(means)) / mean_of_means
        if drift_pct > _SLEEP_PHASE_DRIFT_PCT:
            return None
    else:
        drift_pct = 0.0

    std_score = (_SLEEP_HOURS_STABLE_STD - worst_std) / _SLEEP_HOURS_STABLE_STD
    drift_score = (_SLEEP_PHASE_DRIFT_PCT - drift_pct) / _SLEEP_PHASE_DRIFT_PCT
    strength = min(1.0, (std_score + drift_score) / 2.0 + 0.3)
    # Concrete anchor (Hướng B): name the sleep behaviour SPECIFICALLY rather
    # than a generic "routine". Differentiates from cross-module / cross-phase
    # stability patterns which paraphrase to the same vocabulary.
    sleep_steadiness = _label_timing_std_minutes(worst_std * 60)
    return InsightCandidate(
        pattern_family="B",
        pattern_type="consistent_routine",
        tentative_category="Great Job",
        phase_focus="all",
        raw_template_key="consistent_routine",
        raw_signal={
            "subject": "your sleep hours from night to night",
            "scope": "sleep_duration",
            "stability_label": "stable",
            "sleep_steadiness_label": sleep_steadiness,
            "max_phase_sleep_std": round(worst_std, 3),
            "phase_mean_drift_pct": round(drift_pct, 3),
        },
        signal_strength=round(strength, 3),
        cross_module_impact=0.25,
    )


def _irregular_timing(
    snapshot: MonthlySnapshot, phases: List[PhaseMetrics]
) -> Optional[InsightCandidate]:
    """Rules §3.B row 3: 'High variation in daily schedule' → Opportunity."""
    stds = [p.h_sleep_hours_std for p in phases if p.h_sleep_hours_std is not None]
    worst_std = max(stds) if stds else 0.0
    overall_std = snapshot.overall_day_score_std or 0.0
    if worst_std < _SLEEP_HOURS_NOISY_STD and overall_std < _OVERALL_NOISY_STD:
        return None

    strength = min(
        1.0,
        max(
            (worst_std - _SLEEP_HOURS_NOISY_STD) / _SLEEP_HOURS_NOISY_STD,
            (overall_std - _OVERALL_NOISY_STD) / _OVERALL_NOISY_STD,
        )
        + 0.3,
    )
    # Concrete anchor (Hướng B): name WHICH metric drives the variability
    # (sleep hours vs daily score) so the LLM doesn't paraphrase this and
    # consistent_routine into the same "schedule" vocabulary.
    source = _pick_variability_source(phases, snapshot)
    subject_map = {
        "sleep_hours": "your sleep hours from night to night",
        "daily_activity": "your day-to-day activity",
        "both": "both your sleep hours and your day-to-day activity",
    }
    return InsightCandidate(
        pattern_family="B",
        pattern_type="irregular_timing",
        tentative_category="Opportunity",
        phase_focus="all",
        raw_template_key="irregular_timing",
        raw_signal={
            "subject": subject_map.get(source or "", "daily schedule"),
            "variability_source": source or "unknown",
            "variability_label": _label_imbalance_std(overall_std),
            "improvement_direction": "tighter night-to-night consistency",
            "max_phase_sleep_std": round(worst_std, 3),
            "overall_day_score_std": round(overall_std, 3),
        },
        signal_strength=round(strength, 3),
        cross_module_impact=0.25,
    )


# =============================================================================
# Family C — Sequence detectors (3)
# =============================================================================


def _detect_family_c(
    snapshot: MonthlySnapshot, phases: List[PhaseMetrics]
) -> List[InsightCandidate]:
    out: List[InsightCandidate] = []
    out.extend(_sleep_work_chain(snapshot, phases))
    c2 = _workout_mood_chain(snapshot, phases)
    if c2:
        out.append(c2)
    c3 = _busy_day_spending(snapshot, phases)
    if c3:
        out.append(c3)
    return out


def _sleep_work_chain(
    snapshot: MonthlySnapshot, phases: List[PhaseMetrics]
) -> List[InsightCandidate]:
    """Rules §3.C row 1: 'Poor sleep → high workload → low activity' → Need Attention.

    Two strategies:
      1. Strong negative work×sleep correlation (2-module)
      2. Phase-deterioration: sleep drop + work rise + activity drop p1→p3 (3-module)
    """
    out: List[InsightCandidate] = []

    corr = snapshot.corr_work_sleep
    if corr is not None and corr <= _NEG_CORR_THRESHOLD:
        strength = min(1.0, (abs(corr) - 0.5) / 0.5 + 0.4)
        out.append(
            InsightCandidate(
                pattern_family="C",
                pattern_type="sleep_work_chain",
                tentative_category="Need Attention",
                phase_focus="cross",
                raw_template_key="sleep_work_chain",
                raw_signal={
                    "subject": "the link between your workload and sleep",
                    "scope": "month_long_work_sleep_tradeoff",
                    "direction_unambiguous": "moved together negatively",
                    "time_window_human": "across the month",
                    "improvement_direction": "easing the workload so sleep can recover",
                    "corr_work_sleep": corr,
                    "trigger": "correlation",
                },
                signal_strength=round(strength, 3),
                cross_module_impact=0.75,
            )
        )

    p1, _, p3 = phases
    if p1.days_with_data >= 3 and p3.days_with_data >= 3:
        d_sleep = pct_delta(p3.h_sleep_hours_avg, p1.h_sleep_hours_avg)
        d_work = pct_delta(p3.p_meeting_minutes_avg, p1.p_meeting_minutes_avg)
        d_act = pct_delta(p3.h_active_minutes_avg, p1.h_active_minutes_avg)
        if (
            d_sleep is not None
            and d_sleep <= _SLEEP_DROP_PCT
            and d_work is not None
            and d_work >= _WORK_RISE_PCT
            and d_act is not None
            and d_act <= _ACTIVITY_DROP_PCT
        ):
            mean_abs_delta = (abs(d_sleep) + abs(d_work) + abs(d_act)) / 3.0
            strength = min(1.0, mean_abs_delta / 0.50)
            out.append(
                InsightCandidate(
                    pattern_family="C",
                    pattern_type="sleep_work_chain",
                    tentative_category="Need Attention",
                    phase_focus="3",
                    raw_template_key="sleep_work_chain",
                    raw_signal={
                        "subject": "the sleep, workload, and activity chain",
                        "scope": "phase1_to_phase3_chain_deterioration",
                        "direction_unambiguous": "sleep and activity dropped while work climbed",
                        "time_window_human": "in the closing stretch",
                        "comparison_anchor": "compared to the opening weeks",
                        "improvement_direction": "protecting sleep and movement as work intensifies",
                        "d_sleep": d_sleep,
                        "d_work": d_work,
                        "d_act": d_act,
                        "trigger": "phase_deterioration",
                    },
                    signal_strength=round(strength, 3),
                    cross_module_impact=0.75,
                )
            )

    return out


def _workout_mood_chain(
    snapshot: MonthlySnapshot, phases: List[PhaseMetrics]
) -> Optional[InsightCandidate]:
    """Rules §3.C row 2: 'Workout → mood → productivity' → Great Job (3-module).

    Both legs scored continuously from daily-level Pearson correlations and
    AND-combined via min() — the chain only holds when both legs hold. The
    historical cutoffs (mood ≥ 0.5, productivity ≥ 0.30) become anchor points
    on the gradient rather than binary gates (Feedback §5):

      - mood leg     : linear, peaks at corr_workout_mood = 1.0
      - productivity : linear normalised by 0.6 (matches the looser noise
                        profile of bounded 0-100 productivity scores).
    """
    corr_mood = snapshot.corr_workout_mood
    corr_prod = snapshot.corr_workout_productivity
    if corr_mood is None or corr_prod is None or corr_mood <= 0 or corr_prod <= 0:
        return None
    mood_leg = max(0.0, min(1.0, corr_mood))
    prod_leg = max(0.0, min(1.0, corr_prod / 0.6))
    strength = min(mood_leg, prod_leg)
    if strength < _GRADIENT_NOISE_FLOOR:
        return None
    return InsightCandidate(
        pattern_family="C",
        pattern_type="workout_mood_chain",
        tentative_category="Great Job",
        phase_focus="all",
        raw_template_key="workout_mood_chain",
        raw_signal={
            "subject": "the chain from workouts to mood to productivity",
            "scope": "month_long_workout_mood_productivity_chain",
            "direction_unambiguous": "moved together positively",
            "time_window_human": "across the month",
            "corr_workout_mood": corr_mood,
            "corr_workout_productivity": corr_prod,
        },
        signal_strength=round(strength, 3),
        cross_module_impact=0.75,
    )


def _busy_day_spending(
    snapshot: MonthlySnapshot, phases: List[PhaseMetrics]
) -> Optional[InsightCandidate]:
    """Rules §3.C row 3: 'Busy day → higher spending' → Opportunity.

    Uses lower correlation than F.2 event_driven_spending so the two don't
    fire on the same metric — fires on a *mild* positive busy↔spend link.
    """
    if snapshot.days_with_data < 7:
        return None

    total_events = snapshot.p_total_events_sum
    days = snapshot.days_with_data
    if total_events is None or days == 0:
        return None
    events_per_day = total_events / days
    if events_per_day < _BUSY_EVENTS_PER_DAY:
        return None

    corr = snapshot.corr_work_spend
    if corr is None or not (_BUSY_SPEND_CORR <= corr < _BUSY_SPEND_CORR_CEILING):
        return None

    strength = min(1.0, (corr - _BUSY_SPEND_CORR) / 0.30 + 0.3)
    return InsightCandidate(
        pattern_family="C",
        pattern_type="busy_day_spending",
        tentative_category="Opportunity",
        phase_focus="all",
        raw_template_key="busy_day_spending",
        raw_signal={
            "subject": "the link between busy days and spending",
            "scope": "month_long_busy_day_spend_link",
            "direction_unambiguous": "edged up together",
            "time_window_human": "across the month",
            "improvement_direction": "planning ahead for busy days so spending stays steady",
            "events_per_day": round(events_per_day, 2),
            "corr_work_spend": corr,
        },
        signal_strength=round(strength, 3),
        cross_module_impact=0.50,
    )


# =============================================================================
# Family D — Goal vs Reality detectors (3)
# =============================================================================


def _detect_family_d(
    snapshot: MonthlySnapshot, phases: List[PhaseMetrics]
) -> List[InsightCandidate]:
    """Pick exactly one of D.1 / D.2 / D.3 per month — strongest match wins."""
    out: List[InsightCandidate] = []

    if snapshot.days_with_data >= 7:
        alignment = _collect_alignment(snapshot)
        if len(alignment) >= 2:
            aligned = [m for m, v in alignment.items() if v >= _ALIGNED_FLOOR]
            misaligned = [m for m, v in alignment.items() if v <= _MISALIGNED_CEILING]

            if len(aligned) == len(alignment) and len(alignment) >= 2:
                c = _consistent_goal_alignment(alignment)
            elif misaligned and len(misaligned) >= 2:
                c = _misalignment(alignment, misaligned)
            elif aligned and not misaligned:
                c = _partial_alignment(alignment, aligned)
            elif misaligned:
                c = _misalignment(alignment, misaligned)
            elif aligned:
                c = _partial_alignment(alignment, aligned)
            else:
                c = None
            if c:
                out.append(c)

    return out


def _collect_alignment(snapshot: MonthlySnapshot) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if snapshot.goal_alignment_health_pct is not None:
        out["health"] = max(0.0, min(1.5, snapshot.goal_alignment_health_pct / 100.0))
    if snapshot.goal_alignment_productivity_pct is not None:
        out["productivity"] = max(
            0.0, min(1.5, snapshot.goal_alignment_productivity_pct / 100.0)
        )
    if snapshot.goal_alignment_finance_pct is not None:
        out["finance"] = max(0.0, min(1.5, snapshot.goal_alignment_finance_pct / 100.0))
    return out


def _consistent_goal_alignment(
    alignment: Dict[str, float],
) -> Optional[InsightCandidate]:
    """Rules §3.D row 1: 'Actual ≈ goal across modules' → Great Job."""
    avg = sum(alignment.values()) / len(alignment)
    strength = min(1.0, (avg - _ALIGNED_FLOOR) / 0.20 + 0.5)
    return InsightCandidate(
        pattern_family="D",
        pattern_type="consistent_goal_alignment",
        tentative_category="Great Job",
        phase_focus="all",
        raw_template_key="consistent_goal_alignment",
        raw_signal={
            "alignment_label": _label_alignment_pct(avg),
            "strong_areas": list(alignment.keys()),
            "subject": "goal progress across health, work, and finance",
            "alignment": {k: round(v, 3) for k, v in alignment.items()},
            "avg_alignment": round(avg, 3),
        },
        signal_strength=round(strength, 3),
        cross_module_impact=min(1.0, 0.25 * len(alignment)),
    )


def _partial_alignment(
    alignment: Dict[str, float], aligned: List[str]
) -> Optional[InsightCandidate]:
    """Rules §3.D row 2: 'One module aligned' → Opportunity."""
    avg = sum(alignment.values()) / len(alignment)
    spread = max(alignment.values()) - min(alignment.values())
    strength = min(1.0, spread + 0.3)
    best_area = max(alignment, key=alignment.get)
    weak_areas = [m for m in alignment if m not in aligned]
    return InsightCandidate(
        pattern_family="D",
        pattern_type="partial_alignment",
        tentative_category="Opportunity",
        phase_focus="all",
        raw_template_key="partial_alignment",
        raw_signal={
            "strong_areas": aligned,
            "weak_areas": weak_areas,
            "best_area": best_area,
            "spread_label": "uneven" if spread < 0.40 else "very_uneven",
            "subject": "goal progress",
            "alignment": {k: round(v, 3) for k, v in alignment.items()},
            "aligned_modules": aligned,
            "avg_alignment": round(avg, 3),
        },
        signal_strength=round(strength, 3),
        cross_module_impact=min(1.0, 0.25 * len(alignment)),
    )


def _misalignment(
    alignment: Dict[str, float], misaligned: List[str]
) -> Optional[InsightCandidate]:
    """Rules §3.D row 3: 'Large gap vs goals' → Need Attention."""
    avg = sum(alignment.values()) / len(alignment)
    worst = min(alignment.values())
    worst_area = min(alignment, key=alignment.get)
    strong_areas = [m for m in alignment if m not in misaligned]
    strength = min(1.0, (_MISALIGNED_CEILING - worst) / _MISALIGNED_CEILING + 0.4)
    return InsightCandidate(
        pattern_family="D",
        pattern_type="misalignment",
        tentative_category="Need Attention",
        phase_focus="all",
        raw_template_key="misalignment",
        raw_signal={
            "weak_areas": misaligned,
            "strong_areas": strong_areas,
            "worst_area": worst_area,
            "alignment_label": _label_alignment_pct(worst),
            "subject": "goal progress",
            "alignment": {k: round(v, 3) for k, v in alignment.items()},
            "misaligned_modules": misaligned,
            "avg_alignment": round(avg, 3),
        },
        signal_strength=round(strength, 3),
        cross_module_impact=min(1.0, 0.25 * len(alignment)),
    )


# =============================================================================
# Family E — Location & Travel (stub, blocked by data)
# =============================================================================


def _detect_family_e(
    snapshot: MonthlySnapshot, phases: List[PhaseMetrics]
) -> List[InsightCandidate]:
    """Rules §3.E E.1/E.2.

    NOT IMPLEMENTED. ``DailySnapshot`` has no ``location`` /
    ``timezone_offset_min`` / ``is_travel_day``. When BE exposes those:
      E.1 travel_disruption  → Need Attention (timezone shift + sleep noisy)
      E.2 adaptive_behaviour → Great Job (travel days ≥ 3 + stable metrics)
    """
    return []


# =============================================================================
# Family F — Finance-context detectors (3)
# =============================================================================


def _detect_family_f(
    snapshot: MonthlySnapshot, phases: List[PhaseMetrics]
) -> List[InsightCandidate]:
    out: List[InsightCandidate] = []
    out.extend(_stress_linked_spending(snapshot, phases))
    f2 = _event_driven_spending(snapshot, phases)
    if f2:
        out.append(f2)
    f3 = _controlled_spending(snapshot, phases)
    if f3:
        out.append(f3)
    return out


def _stress_linked_spending(
    snapshot: MonthlySnapshot, phases: List[PhaseMetrics]
) -> List[InsightCandidate]:
    """Rules §3.F row 3: 'Work ↑ + spend ↑' → Need Attention."""
    out: List[InsightCandidate] = []

    corr = snapshot.corr_work_spend
    if corr is not None and corr >= _CORR_STRESS_SPEND:
        strength = min(1.0, (corr - 0.5) / 0.5 + 0.4)
        out.append(
            InsightCandidate(
                pattern_family="F",
                pattern_type="stress_linked_spending",
                tentative_category="Need Attention",
                phase_focus="cross",
                raw_template_key="stress_linked_spending",
                raw_signal={
                    "subject": "the link between workload and spending",
                    "scope": "month_long_work_spend_link",
                    "direction_unambiguous": "rose together",
                    "time_window_human": "across the month",
                    "improvement_direction": "interrupting the busy-day to spending reflex",
                    "corr_work_spend": corr,
                    "trigger": "correlation",
                },
                signal_strength=round(strength, 3),
                cross_module_impact=0.50,
            )
        )

    p1, _, p3 = phases
    if p1.days_with_data >= 3 and p3.days_with_data >= 3:
        d_spend = pct_delta(p3.f_total_expense_sum, p1.f_total_expense_sum)
        d_work = pct_delta(p3.p_meeting_minutes_avg, p1.p_meeting_minutes_avg)
        if (
            d_spend is not None
            and d_spend >= _SPEND_RISE_PCT
            and d_work is not None
            and d_work >= _WORK_RISE_PCT_FOR_SPEND
        ):
            mean_abs_delta = (abs(d_spend) + abs(d_work)) / 2.0
            strength = min(1.0, mean_abs_delta / 0.50)
            out.append(
                InsightCandidate(
                    pattern_family="F",
                    pattern_type="stress_linked_spending",
                    tentative_category="Need Attention",
                    phase_focus="3",
                    raw_template_key="stress_linked_spending",
                    raw_signal={
                        "subject": "your workload and spending",
                        "scope": "phase1_to_phase3_work_spend_rise",
                        "direction_unambiguous": "climbed together",
                        "time_window_human": "in the closing stretch",
                        "comparison_anchor": "compared to the opening weeks",
                        "improvement_direction": "keeping spending steady even when work spikes",
                        "d_spend": d_spend,
                        "d_work": d_work,
                        "trigger": "phase_rise",
                    },
                    signal_strength=round(strength, 3),
                    cross_module_impact=0.50,
                )
            )

    return out


def _event_driven_spending(
    snapshot: MonthlySnapshot, phases: List[PhaseMetrics]
) -> Optional[InsightCandidate]:
    """Rules §3.F row 1: 'Spend spikes on event days' → Opportunity."""
    corr = snapshot.corr_work_spend
    if corr is None:
        return None
    if not (_CORR_EVENT_SPEND <= corr < _CORR_STRESS_SPEND):
        return None

    strength = min(1.0, (corr - _CORR_EVENT_SPEND) / 0.30 + 0.3)
    return InsightCandidate(
        pattern_family="F",
        pattern_type="event_driven_spending",
        tentative_category="Opportunity",
        phase_focus="all",
        raw_template_key="event_driven_spending",
        raw_signal={
            "subject": "spending on event-heavy days",
            "scope": "month_long_event_day_spend_spikes",
            "direction_unambiguous": "ticked up alongside busy days",
            "time_window_human": "across the month",
            "improvement_direction": "smoothing out event-day spending",
            "corr_work_spend": corr,
        },
        signal_strength=round(strength, 3),
        cross_module_impact=0.50,
    )


def _controlled_spending(
    snapshot: MonthlySnapshot, phases: List[PhaseMetrics]
) -> Optional[InsightCandidate]:
    """Rules §3.F row 2: 'Stable spend despite high activity' → Great Job."""
    avg_mtg = snapshot.p_meeting_minutes_avg
    if avg_mtg is None or avg_mtg < _CONTROLLED_MEETING_MIN:
        return None
    corr = snapshot.corr_work_spend
    if corr is None or abs(corr) > _CORR_CONTROLLED_CEILING:
        return None

    strength = (
        0.5 + (_CORR_CONTROLLED_CEILING - abs(corr)) / _CORR_CONTROLLED_CEILING * 0.3
    )
    return InsightCandidate(
        pattern_family="F",
        pattern_type="controlled_spending",
        tentative_category="Great Job",
        phase_focus="all",
        raw_template_key="controlled_spending",
        raw_signal={
            "subject": "your spending while the workload stayed high",
            "scope": "month_long_spend_independent_of_work",
            "direction_unambiguous": "stayed steady",
            "time_window_human": "across the month",
            "p_meeting_minutes_avg": avg_mtg,
            "corr_work_spend": corr,
        },
        signal_strength=round(strength, 3),
        cross_module_impact=0.50,
    )


# =============================================================================
# Family X — Cross-phase / cross-module / final-day synthesis detectors
# =============================================================================


def _detect_family_x(
    snapshot: MonthlySnapshot, phases: List[PhaseMetrics]
) -> List[InsightCandidate]:
    out: List[InsightCandidate] = []
    for fn in (
        _stable_rhythm,
        _stable_rhythm_p2,
        _declining_momentum,
        _building_momentum,
        _late_concentration,
        _skewed_allocation,
        _balanced_start,
        _early_low_activity,
        # Added per Rules §1 Phase 2 / Phase 3 (event-driven gap)
        _sustained_balance,
        _growing_imbalance,
        _cross_module_imbalance,
        _irregular_finish,
        _catchup_pattern,
        _strong_finish,
        _finance_imbalance,
        _work_health_tradeoff_corr,
        _finance_work_link,
        # Phase variance + cross-module gaps (Rules §1)
        _early_variability,
        _increasing_variability,
        _cross_module_stability,
        _controlled_increase,
        _stress_pattern,
        # New DDL-field detector (mood intraday range)
        _mood_volatility,
        # Rules §1 Final-Day synthesis (rows 1-3; row 4 needs baseline, row 5 already covered)
        _dominant_pattern_final_day,
        _tradeoff_summary_final_day,
        _stability_summary_final_day,
    ):
        c = fn(snapshot, phases)
        if c:
            out.append(c)
    return out


def _stable_rhythm_p2(
    snapshot: MonthlySnapshot, phases: List[PhaseMetrics]
) -> Optional[InsightCandidate]:
    """Rules §1 Phase 2 row 4: 'Stable rhythm — Low variance across modules' → Great Job.

    Cross-module measure WITHIN Phase 2 (distinct from ``_stable_rhythm`` which
    measures across phases). Picks the 3 module scores (h_health, productivity,
    financial_health) in Phase 2 and checks their std-dev — if low, the user's
    routine is "steady across different areas" mid-month per spec text.

    Mood is excluded for consistency with ``_module_balance`` /
    ``_phase_balance`` — no balance pattern in Rules references mood.
    All 3 module scores already live on the same 0-100 scale, so no per-field
    rescaling is needed.
    """
    p2 = phases[1]
    if p2.days_with_data < 5:
        return None
    module_scores: List[float] = []
    if p2.h_health_score_avg is not None:
        module_scores.append(float(p2.h_health_score_avg))
    if p2.productivity_score_avg is not None:
        module_scores.append(float(p2.productivity_score_avg))
    if p2.financial_health_score_avg is not None:
        module_scores.append(float(p2.financial_health_score_avg))
    if len(module_scores) < _MIN_MODULES_FOR_BALANCE:
        return None
    mean = sum(module_scores) / len(module_scores)
    variance = sum((x - mean) ** 2 for x in module_scores) / len(module_scores)
    std = variance**0.5
    if std > _STABLE_RHYTHM_P2_MAX_STD:
        return None
    strength = min(
        1.0,
        (_STABLE_RHYTHM_P2_MAX_STD - std) / _STABLE_RHYTHM_P2_MAX_STD + 0.5,
    )
    # Concrete anchor: stable_rhythm_p2 is about MID-MONTH cross-module
    # spread — distinguish it from stable_rhythm (cross-phase) by naming the
    # scope explicitly.
    return InsightCandidate(
        pattern_family="X",
        pattern_type="stable_rhythm_p2",
        tentative_category="Great Job",
        phase_focus="2",
        raw_template_key="stable_rhythm_p2",
        raw_signal={
            "subject": "how evenly your health, work, and finance moved together",
            "scope": "mid_month_cross_module",
            "stability_label": "even",
            "phase2_module_std": round(std, 2),
            "modules_compared": len(module_scores),
        },
        signal_strength=round(strength, 3),
        cross_module_impact=0.50,
    )


def _stable_rhythm(
    snapshot: MonthlySnapshot, phases: List[PhaseMetrics]
) -> Optional[InsightCandidate]:
    """Rules §1 Phase 3 row 2: 'Sustained performance' → Great Job."""
    scores = [
        p.overall_day_score_avg for p in phases if p.overall_day_score_avg is not None
    ]
    if len(scores) != 3:
        return None
    lo, hi = min(scores), max(scores)
    if lo == 0:
        return None
    spread = (hi - lo) / lo
    if spread > _STABLE_PCT:
        return None

    strength = min(1.0, (_STABLE_PCT - spread) / _STABLE_PCT + 0.5)
    # Concrete anchor: stable_rhythm is about ACROSS-PHASE consistency (P1→P2→P3
    # stays similar). Distinct from stable_rhythm_p2 (cross-module within P2).
    return InsightCandidate(
        pattern_family="X",
        pattern_type="stable_rhythm",
        tentative_category="Great Job",
        phase_focus="cross",
        raw_template_key="stable_rhythm",
        raw_signal={
            "subject": "the day-to-day activity from start to end of the month",
            "scope": "across_phases",
            "stability_label": "consistent",
            "phase_score_spread_pct": round(spread, 3),
        },
        signal_strength=round(strength, 3),
        cross_module_impact=0.50,
    )


def _declining_momentum(
    snapshot: MonthlySnapshot, phases: List[PhaseMetrics]
) -> Optional[InsightCandidate]:
    """Rules §1 Phase 2 row 1: 'Phase 2 < Phase 1 by >20%' → Need Attention."""
    p1 = phases[0].overall_day_score_avg
    p2 = phases[1].overall_day_score_avg
    delta = pct_delta(p2, p1)
    if delta is None or delta > _TREND_DECLINE:
        return None

    strength = min(1.0, abs(delta) / 0.50)
    return InsightCandidate(
        pattern_family="X",
        pattern_type="declining_momentum",
        tentative_category="Need Attention",
        phase_focus="2",
        raw_template_key="declining_momentum",
        raw_signal={
            "subject": "your day-to-day activity",
            "scope": "phase1_to_phase2_decline",
            "direction_unambiguous": "slipped lower",
            "time_window_human": "by mid-month",
            "comparison_anchor": "compared to the opening weeks",
            "improvement_direction": "rebuilding momentum back toward the early-month level",
            "phase2_vs_phase1_delta": delta,
        },
        signal_strength=round(strength, 3),
        cross_module_impact=0.50,
    )


def _building_momentum(
    snapshot: MonthlySnapshot, phases: List[PhaseMetrics]
) -> Optional[InsightCandidate]:
    """Rules §1 Phase 2 row 2: 'Phase 2 > Phase 1' → Opportunity."""
    p1 = phases[0].overall_day_score_avg
    p2 = phases[1].overall_day_score_avg
    delta = pct_delta(p2, p1)
    if delta is None or delta < _TREND_BUILD:
        return None

    strength = min(1.0, delta / 0.50 + 0.3)
    return InsightCandidate(
        pattern_family="X",
        pattern_type="building_momentum",
        tentative_category="Opportunity",
        phase_focus="2",
        raw_template_key="building_momentum",
        raw_signal={
            "subject": "your day-to-day activity",
            "scope": "phase1_to_phase2_uptick",
            "direction_unambiguous": "climbed higher",
            "time_window_human": "by mid-month",
            "comparison_anchor": "compared to the opening weeks",
            "phase2_vs_phase1_delta": delta,
        },
        signal_strength=round(strength, 3),
        cross_module_impact=0.50,
    )


def _late_concentration(
    snapshot: MonthlySnapshot, phases: List[PhaseMetrics]
) -> Optional[InsightCandidate]:
    """Rules §1 Phase 3 row 1: 'Phase 3 highest' → Opportunity.

    Excludes the P2-dip shape (P1 > P2) so ``_catchup_pattern`` claims that
    territory exclusively. Fires on the smoother P1 ≤ P2 < P3 case where the
    final third stands out without a mid-month slump.
    """
    s1 = phases[0].overall_day_score_avg
    s2 = phases[1].overall_day_score_avg
    s3 = phases[2].overall_day_score_avg
    if s1 is None or s2 is None or s3 is None:
        return None
    if not (s3 > s1 and s3 > s2):
        return None
    # Exclude catch-up territory: P2 must NOT be a dip vs P1.
    if s2 < s1:
        return None

    delta = pct_delta(s3, max(s1, s2))
    if delta is None or delta < _LATE_CONCENTRATION_PCT:
        return None

    strength = min(1.0, delta / 0.50 + 0.3)
    return InsightCandidate(
        pattern_family="X",
        pattern_type="late_concentration",
        tentative_category="Opportunity",
        phase_focus="3",
        raw_template_key="late_concentration",
        raw_signal={
            "subject": "your day-to-day activity",
            "scope": "late_month_peak",
            "direction_unambiguous": "rose to its highest of the month",
            "time_window_human": "in the closing stretch",
            "comparison_anchor": "compared to early and mid-month",
            "p3_vs_max_p1p2_delta": delta,
        },
        signal_strength=round(strength, 3),
        cross_module_impact=0.50,
    )


def _skewed_allocation(
    snapshot: MonthlySnapshot, phases: List[PhaseMetrics]
) -> Optional[InsightCandidate]:
    """Rules §1 Phase 1 row 4: 'One module >30% higher than others' → Opportunity.

    Gated to the early-month window (snapshot.days_with_data ≤ _EARLY_DAYS_CAP)
    so it doesn't overlap with ``_cross_module_imbalance`` (P3 row 8) or
    ``_dominant_pattern_final_day`` (Final-Day row 1), which surface the same
    imbalance signal later in the month with different framing.
    """
    if snapshot.days_with_data > _EARLY_DAYS_CAP:
        return None
    excess = snapshot.module_max_excess_pct
    if excess is None or excess < _SKEW_EXCESS_PCT:
        return None

    strength = min(1.0, (excess - _SKEW_EXCESS_PCT) / _SKEW_EXCESS_PCT + 0.4)
    return InsightCandidate(
        pattern_family="X",
        pattern_type="skewed_allocation",
        tentative_category="Opportunity",
        phase_focus="1",
        raw_template_key="skewed_allocation",
        raw_signal={
            "subject": "how attention is split across health, work, and finance",
            "scope": "early_month_one_module_dominant",
            "time_window_human": "early in the month",
            "improvement_direction": "rebalancing time across health, work, and finance",
            "module_max_excess_pct": excess,
            "balance_score": snapshot.balance_score,
            "module_scores": snapshot.module_scores,
            "area": snapshot.most_concentrated_area or "unknown",
        },
        signal_strength=round(strength, 3),
        cross_module_impact=1.0,
    )


def _balanced_start(
    snapshot: MonthlySnapshot, phases: List[PhaseMetrics]
) -> Optional[InsightCandidate]:
    """Rules §1 Phase 1 row 5: 'All modules within ±15% range' → Great Job.

    Phase-1-scoped counterpart to ``_skewed_allocation``: where skewed fires
    on excess ≥30%, balanced fires when excess ≤15%. Uses ``module_max_excess_pct``
    intentionally — spec rows 4-5 of P1 use the "module excess" metric, while
    Phase 2 rows 6-7 use a different "balance score" metric (std-based). Both
    are valid per spec; do not unify.

    Gated to the early window (``days_with_data ≤ _EARLY_DAYS_CAP``) so it
    doesn't overlap with ``_sustained_balance`` (P2) later in the month.
    """
    p1 = phases[0]
    if p1.days_with_data < 3:
        return None
    if snapshot.days_with_data > _EARLY_DAYS_CAP:
        return None
    excess = snapshot.module_max_excess_pct
    if excess is None or excess > _BALANCED_START_MAX_EXCESS:
        return None
    # Strength inversely proportional to excess: 0 excess → ~1.0, 15% → ~0.5
    strength = min(
        1.0, (_BALANCED_START_MAX_EXCESS - excess) / _BALANCED_START_MAX_EXCESS + 0.5
    )
    return InsightCandidate(
        pattern_family="X",
        pattern_type="balanced_start",
        tentative_category="Great Job",
        phase_focus="1",
        raw_template_key="balanced_start",
        raw_signal={
            "subject": "the balance across health, work, and finance",
            "scope": "early_month_modules_in_balance",
            "time_window_human": "early in the month",
            "module_max_excess_pct": excess,
            "balance_score": snapshot.balance_score,
        },
        signal_strength=round(strength, 3),
        cross_module_impact=1.0,
    )


def _early_low_activity(
    snapshot: MonthlySnapshot, phases: List[PhaseMetrics]
) -> Optional[InsightCandidate]:
    """Rules §1 Phase 1 row 8: 'Flat start — All modules low vs 90d baseline' → Opportunity."""
    p1 = phases[0]
    if p1.days_with_data < 3:
        return None
    score = p1.overall_day_score_avg
    if score is None or score >= _EARLY_LOW_SCORE:
        return None
    if snapshot.days_with_data > _EARLY_DAYS_CAP:
        return None

    strength = min(1.0, (_EARLY_LOW_SCORE - score) / _EARLY_LOW_SCORE + 0.3)
    return InsightCandidate(
        pattern_family="X",
        pattern_type="early_low_activity",
        tentative_category="Opportunity",
        phase_focus="1",
        raw_template_key="early_low_activity",
        raw_signal={
            "subject": "your day-to-day activity in the opening weeks",
            "scope": "early_month_flat_start",
            "time_window_human": "early in the month",
            "improvement_direction": "lifting activity across health, work, and finance to start the month stronger",
            "phase1_overall_avg": score,
        },
        signal_strength=round(strength, 3),
        cross_module_impact=1.0,
    )


def _sustained_balance(
    snapshot: MonthlySnapshot, phases: List[PhaseMetrics]
) -> Optional[InsightCandidate]:
    """Rules §1 Phase 2 row 6: 'Balance score across Phase 1 & 2 remains above 75%'.

    balance_score = mean(h_health_score, productivity_score,
    financial_health_score) on 0..100 scale per spec. Per-phase mean is
    averaged across P1 + P2; pattern fires when avg_balance ≥ 75. Because the
    metric measures level (not just spread), this no longer false-fires for
    users who are "balanced low" across all modules — their mean stays low.
    """
    p1 = phases[0]
    p2 = phases[1]
    if p1.days_with_data < 3 or p2.days_with_data < 3:
        return None
    bal_p1 = _phase_balance(p1)
    bal_p2 = _phase_balance(p2)
    if bal_p1 is None or bal_p2 is None:
        return None
    avg_balance = (bal_p1 + bal_p2) / 2.0
    if avg_balance < _SUSTAINED_BALANCE_MIN_AVG:
        return None
    # Scale 0..100; headroom above the 75-floor is 25 points → normalize.
    strength = min(1.0, (avg_balance - _SUSTAINED_BALANCE_MIN_AVG) / 25.0 + 0.5)
    # Concrete anchors (Hướng B): name the leading module + companions so the
    # LLM can write "your health side held strong, with work and finance both
    # solid" rather than a generic "activity stayed stable".
    top_module = _pick_top_module(snapshot.module_scores)
    supporting = _pick_supporting_modules(snapshot.module_scores, top_module)
    return InsightCandidate(
        pattern_family="X",
        pattern_type="sustained_balance",
        tentative_category="Great Job",
        phase_focus="2",
        raw_template_key="sustained_balance",
        raw_signal={
            "level_label": _label_position_0_100(avg_balance),
            "balance_strength_label": _label_balance_strength(avg_balance),
            "top_module": top_module,
            "supporting_modules": supporting,
            "subject": "level across health, work, and finance",
            # Internal numbers retained for scoring/audit; strip layer hides
            # them from LLM via _INTERNAL_RAW_SIGNAL_KEYS.
            "phase1_balance": round(bal_p1, 2),
            "phase2_balance": round(bal_p2, 2),
            "avg_balance": round(avg_balance, 2),
        },
        signal_strength=round(strength, 3),
        cross_module_impact=1.0,
    )


def _growing_imbalance(
    snapshot: MonthlySnapshot, phases: List[PhaseMetrics]
) -> Optional[InsightCandidate]:
    """Rules §1 Phase 2 row 7: 'Balance score increasing vs Phase 1' → Opportunity.

    With the mean-based balance_score, spec wording is reinterpreted: fires
    when Phase 2 balance DROPS from Phase 1 by ≥ ``_PHASE_BALANCE_DROP_MIN``
    (level declining = the "imbalance growing" signal the rule was targeting).
    Spread-based imbalance is handled separately by ``_cross_module_imbalance``
    via ``_monthly_imbalance_std``.
    """
    p1 = phases[0]
    p2 = phases[1]
    if p1.days_with_data < 3 or p2.days_with_data < 3:
        return None
    bal_p1 = _phase_balance(p1)
    bal_p2 = _phase_balance(p2)
    if bal_p1 is None or bal_p2 is None:
        return None
    balance_drop = bal_p1 - bal_p2  # positive ⇒ P2 lower than P1 ⇒ level decline
    if balance_drop < _PHASE_BALANCE_DROP_MIN:
        return None
    # Normalize over a 30-point drop band; floor at 0.4 so even threshold-edge
    # signals carry through to scoring.
    strength = min(1.0, balance_drop / 30.0 + 0.4)
    return InsightCandidate(
        pattern_family="X",
        pattern_type="growing_imbalance",
        tentative_category="Opportunity",
        phase_focus="2",
        raw_template_key="growing_imbalance",
        raw_signal={
            "subject": "health, work, and finance together",
            "scope": "phase1_to_phase2_level_declining",
            "direction_unambiguous": "came down",
            "time_window_human": "by mid-month",
            "comparison_anchor": "compared to the opening weeks",
            "improvement_direction": "lifting health, work, and finance back up together",
            "magnitude_label": _label_balance_drop(balance_drop),
            "direction": "declining",
            "leg": "from early to mid-month",
            "phase1_balance": round(bal_p1, 2),
            "phase2_balance": round(bal_p2, 2),
            "balance_drop": round(balance_drop, 2),
        },
        signal_strength=round(strength, 3),
        cross_module_impact=1.0,
    )


def _cross_module_imbalance(
    snapshot: MonthlySnapshot, phases: List[PhaseMetrics]
) -> Optional[InsightCandidate]:
    """Rules §1 Phase 3 row 9: 'High imbalance — balance score high' → Need Attention.

    With the mean-based balance_score, this pattern intentionally uses a
    SEPARATE SPREAD metric (stddev of the same 3 modules) to detect
    "modules are far apart" — spec wording targets imbalance, not low level.
    Mood is excluded (as in balance_score) per spec scope.
    """
    imbalance_std = _monthly_imbalance_std(snapshot)
    if imbalance_std is None or imbalance_std < _IMBALANCE_HIGH_STD:
        return None
    # Headroom from the threshold to a wide-spread anchor (~30) drives strength.
    strength = min(1.0, (imbalance_std - _IMBALANCE_HIGH_STD) / 15.0 + 0.5)
    # Classify each module as strong vs weak relative to their mean.
    module_scores = snapshot.module_scores or {}
    strong_modules: List[str] = []
    weak_modules: List[str] = []
    if module_scores:
        mean_score = sum(module_scores.values()) / len(module_scores)
        for name, score in module_scores.items():
            if score >= mean_score + 10:
                strong_modules.append(name)
            elif score <= mean_score - 10:
                weak_modules.append(name)
    return InsightCandidate(
        pattern_family="X",
        pattern_type="cross_module_imbalance",
        tentative_category="Need Attention",
        phase_focus="3",
        raw_template_key="cross_module_imbalance",
        raw_signal={
            "subject": "how evenly health, work, and finance moved together",
            "scope": "late_month_module_spread",
            "time_window_human": "in the closing stretch",
            "spread_label": _label_imbalance_std(imbalance_std),
            "strong_areas": strong_modules,
            "weak_areas": weak_modules,
            "dominant_area": snapshot.most_concentrated_area or "unknown",
            "imbalance_std": round(imbalance_std, 2),
            "balance_score": (
                round(snapshot.balance_score, 2)
                if snapshot.balance_score is not None
                else None
            ),
            "area": snapshot.most_concentrated_area or "unknown",
        },
        signal_strength=round(strength, 3),
        cross_module_impact=1.0,
    )


def _irregular_finish(
    snapshot: MonthlySnapshot, phases: List[PhaseMetrics]
) -> Optional[InsightCandidate]:
    """Rules §1 Phase 3 row 8: 'Irregular finish — high day-to-day variance in Phase 3' → Need Attention.

    (Row 3 is owned by ``_strong_finish``; this detector measures Phase-3
    overall_day_score std-dev, i.e. day-to-day swings, not cross-module spread.)
    """
    p3_std = phases[2].overall_day_score_std
    if p3_std is None or p3_std < _IRREGULAR_FINISH_STD:
        return None
    strength = min(1.0, (p3_std - _IRREGULAR_FINISH_STD) / _IRREGULAR_FINISH_STD + 0.5)
    return InsightCandidate(
        pattern_family="X",
        pattern_type="irregular_finish",
        tentative_category="Need Attention",
        phase_focus="3",
        raw_template_key="irregular_finish",
        raw_signal={
            "subject": "your day-to-day activity in the closing stretch",
            "scope": "late_month_day_to_day_variability",
            "time_window_human": "in the closing stretch",
            "improvement_direction": "settling into a steadier day-to-day rhythm",
            "phase3_overall_day_score_std": round(p3_std, 2),
        },
        signal_strength=round(strength, 3),
        cross_module_impact=0.50,
    )


def _catchup_pattern(
    snapshot: MonthlySnapshot, phases: List[PhaseMetrics]
) -> Optional[InsightCandidate]:
    """Rules §1 Phase 3 row 6: 'Late spike after a mid dip — P3 >> P2 after P2 < P1' → Opportunity.

    Distinct from ``_strong_finish``: catch-up requires a Phase 2 dip first.
    Without the dip, falls through so ``_strong_finish`` can claim it as
    Great Job.
    """
    s1 = phases[0].overall_day_score_avg
    s2 = phases[1].overall_day_score_avg
    s3 = phases[2].overall_day_score_avg
    if s1 is None or s2 is None or s3 is None:
        return None
    # Need a P2 dip vs P1, then P3 catch-up vs P2
    if s2 >= s1:
        return None
    delta = pct_delta(s3, s2)
    if delta is None or delta < _CATCHUP_PCT:
        return None
    strength = min(1.0, delta / 0.50 + 0.3)
    return InsightCandidate(
        pattern_family="X",
        pattern_type="catchup_pattern",
        tentative_category="Opportunity",
        phase_focus="3",
        raw_template_key="catchup_pattern",
        raw_signal={
            "subject": "your day-to-day activity",
            "scope": "late_month_catchup_after_mid_dip",
            "direction_unambiguous": "bounced back after a mid-month dip",
            "time_window_human": "in the closing stretch",
            "comparison_anchor": "compared to mid-month",
            "p2_vs_p1_delta": round(pct_delta(s2, s1) or 0.0, 3),
            "p3_vs_p2_delta": round(delta, 3),
        },
        signal_strength=round(strength, 3),
        cross_module_impact=0.50,
    )


def _strong_finish(
    snapshot: MonthlySnapshot, phases: List[PhaseMetrics]
) -> Optional[InsightCandidate]:
    """Rules §1 Phase 3 row 3: 'Strong finish — P3 > P2 significantly' → Great Job.

    Distinct from ``_catchup_pattern``: strong finish does NOT require a
    P2 dip. Triggers when Phase 2 was at or above Phase 1 (i.e. no dip)
    AND Phase 3 still outperforms Phase 2 by ≥ ``_STRONG_FINISH_PCT``.
    """
    s1 = phases[0].overall_day_score_avg
    s2 = phases[1].overall_day_score_avg
    s3 = phases[2].overall_day_score_avg
    if s1 is None or s2 is None or s3 is None:
        return None
    # Exclude catch-up territory: P2 must NOT be a dip vs P1.
    if s2 < s1:
        return None
    delta = pct_delta(s3, s2)
    if delta is None or delta < _STRONG_FINISH_PCT:
        return None
    strength = min(1.0, delta / 0.40 + 0.4)
    return InsightCandidate(
        pattern_family="X",
        pattern_type="strong_finish",
        tentative_category="Great Job",
        phase_focus="3",
        raw_template_key="strong_finish",
        raw_signal={
            "subject": "your day-to-day activity",
            "scope": "late_month_smooth_rise",
            "direction_unambiguous": "rose smoothly into the highest finish of the month",
            "time_window_human": "in the closing stretch",
            "comparison_anchor": "compared to mid-month",
            "p2_vs_p1_delta": round(pct_delta(s2, s1) or 0.0, 3),
            "p3_vs_p2_delta": round(delta, 3),
        },
        signal_strength=round(strength, 3),
        cross_module_impact=0.50,
    )


def _finance_imbalance(
    snapshot: MonthlySnapshot, phases: List[PhaseMetrics]
) -> Optional[InsightCandidate]:
    """Rules §1 Phase 3 row 7: 'Spend spike — Finance ↑ significantly late' → Opportunity.

    Uses per-day spending (``f_total_expense_sum / days_with_data``) so phases
    of unequal length compare fairly. Higher spend per day in Phase 3 vs
    average of Phase 1 / Phase 2 ≥ +20% triggers the insight. Earlier
    revision used ``financial_health_score_avg`` which moves INVERSE to
    spending — that was reversed from spec intent ("spending increased").
    """
    p1, p2, p3 = phases[0], phases[1], phases[2]
    if p1.days_with_data == 0 or p2.days_with_data == 0 or p3.days_with_data == 0:
        return None
    if (
        p1.f_total_expense_sum is None
        or p2.f_total_expense_sum is None
        or p3.f_total_expense_sum is None
    ):
        return None
    e1 = p1.f_total_expense_sum / p1.days_with_data
    e2 = p2.f_total_expense_sum / p2.days_with_data
    e3 = p3.f_total_expense_sum / p3.days_with_data
    baseline = (e1 + e2) / 2.0
    delta = pct_delta(e3, baseline)
    if delta is None or delta < _FINANCE_LATE_DOM_PCT:
        return None
    strength = min(1.0, delta / 0.50 + 0.3)
    return InsightCandidate(
        pattern_family="X",
        pattern_type="finance_imbalance",
        tentative_category="Opportunity",
        phase_focus="3",
        raw_template_key="finance_imbalance",
        raw_signal={
            "subject": "your daily spending in the closing stretch",
            "scope": "late_month_spending_spike",
            "direction_unambiguous": "climbed higher",
            "time_window_human": "in the closing stretch",
            "comparison_anchor": "compared to early and mid-month",
            "improvement_direction": "evening out spending across the rest of the month",
            "phase3_expense_per_day": round(e3, 2),
            "baseline_p1p2_expense_per_day": round(baseline, 2),
            "p3_vs_baseline_delta": round(delta, 3),
        },
        signal_strength=round(strength, 3),
        cross_module_impact=0.75,
    )


def _recovery_rebound(
    snapshot: MonthlySnapshot,
    phases: List[PhaseMetrics],
    baseline: Optional[BaselineContext] = None,
) -> Optional[InsightCandidate]:
    """Rules §1 Phase 2 row 9: 'Rebalancing — Health ↑ after earlier drop' → Opportunity.

    "Earlier drop" must be validated: Phase 1 health must itself have dropped
    relative to either the previous month's full-month health OR the 90d
    baseline by at least ``_RECOVERY_REBOUND_BASELINE_DROP_PCT``. When no
    baseline is available we conservatively fall through (no insight) — using
    Phase 2 > Phase 1 alone would re-label any building momentum as rebound.
    """
    h1 = phases[0].h_health_score_avg
    h2 = phases[1].h_health_score_avg
    if h1 is None or h2 is None:
        return None
    delta = pct_delta(h2, h1)
    if delta is None or delta < _RECOVERY_REBOUND_PCT:
        return None
    # Validate "earlier drop" — Phase 1 must be below baseline meaningfully.
    if baseline is None:
        return None
    prev_h = (
        baseline.prev_month_snapshot.h_health_score_avg
        if baseline.prev_month_snapshot
        else None
    )
    avg_h_90d = baseline.avg_90d.get("h_health_score_avg")
    refs = [r for r in (prev_h, avg_h_90d) if r is not None]
    if not refs:
        return None
    earlier_drop = pct_delta(h1, max(refs))
    if earlier_drop is None or earlier_drop > _RECOVERY_REBOUND_BASELINE_DROP_PCT:
        return None
    strength = min(1.0, delta / 0.50 + 0.3)
    return InsightCandidate(
        pattern_family="X",
        pattern_type="recovery_rebound",
        tentative_category="Opportunity",
        phase_focus="2",
        raw_template_key="recovery_rebound",
        raw_signal={
            "subject": "your health side after an earlier dip",
            "scope": "mid_month_health_rebound",
            "direction_unambiguous": "picked back up",
            "time_window_human": "from the opening weeks into mid-month",
            "comparison_anchor": "compared to last month and the 90-day average",
            "phase1_health_avg": round(h1, 2),
            "phase2_health_avg": round(h2, 2),
            "delta": round(delta, 3),
            "earlier_drop": round(earlier_drop, 3),
        },
        signal_strength=round(strength, 3),
        cross_module_impact=0.75,
    )


def _work_health_tradeoff_corr(
    snapshot: MonthlySnapshot, phases: List[PhaseMetrics]
) -> Optional[InsightCandidate]:
    """Rules §1 Phase 2 row 8: 'Work ↑ AND Health ↓ (corr < -0.5)' → Need Attention.

    ``_NEG_CORR_THRESHOLD`` (-0.5) stays as a documentary anchor: at that
    correlation the new linear formula yields strength = 0.5, matching the
    original cliff value. Weaker negative correlations now scale down to the
    global noise floor instead of being silenced (Feedback §5).
    """
    r = snapshot.corr_work_health
    if r is None or r >= 0:
        return None  # only negative correlations belong to this pattern
    strength = max(0.0, min(1.0, abs(r)))
    if strength < _GRADIENT_NOISE_FLOOR:
        return None
    return InsightCandidate(
        pattern_family="X",
        pattern_type="work_health_tradeoff_corr",
        tentative_category="Need Attention",
        phase_focus="cross",
        raw_template_key="work_health_tradeoff_corr",
        raw_signal={
            "subject": "the link between workload and health",
            "scope": "month_long_work_health_tradeoff",
            "direction_unambiguous": "moved in opposite directions",
            "time_window_human": "across the month",
            "improvement_direction": "protecting health time even when workload climbs",
            "corr_work_health": round(r, 3),
        },
        signal_strength=round(strength, 3),
        cross_module_impact=1.0,
    )


def _finance_work_link(
    snapshot: MonthlySnapshot, phases: List[PhaseMetrics]
) -> Optional[InsightCandidate]:
    """Rules §1 Phase 2 row 10: 'Spend increase with workload — Work ↑ + Spend ↑' → Opportunity (corr > +0.5).

    ``_POS_CORR_THRESHOLD`` (+0.5) is now a documentary anchor — that
    correlation produces strength = 0.5 with the new linear formula.
    Weaker positive correlations decay smoothly toward the global noise
    floor instead of being silenced (Feedback §5).
    """
    r = snapshot.corr_work_spend
    if r is None or r <= 0:
        return None  # only positive correlations belong to this pattern
    strength = max(0.0, min(1.0, r))
    if strength < _GRADIENT_NOISE_FLOOR:
        return None
    return InsightCandidate(
        pattern_family="X",
        pattern_type="finance_work_link",
        tentative_category="Opportunity",
        phase_focus="cross",
        raw_template_key="finance_work_link",
        raw_signal={
            "subject": "the link between workload and spending",
            "scope": "month_long_work_spend_link",
            "direction_unambiguous": "moved together positively",
            "time_window_human": "across the month",
            "improvement_direction": "loosening the workload-to-spending reflex",
            "corr_work_spend": round(r, 3),
        },
        signal_strength=round(strength, 3),
        cross_module_impact=1.0,
    )


def _early_variability(
    snapshot: MonthlySnapshot, phases: List[PhaseMetrics]
) -> Optional[InsightCandidate]:
    """Rules §1 Phase 1 row 6: 'Early variability — High std dev within modules' → Opportunity (early signal)."""
    p1 = phases[0]
    if p1.days_with_data < 4:
        return None
    std = p1.overall_day_score_std
    if std is None or std < _EARLY_VARIABILITY_STD:
        return None
    if snapshot.days_with_data > _EARLY_DAYS_CAP:
        return None
    strength = min(1.0, (std - _EARLY_VARIABILITY_STD) / _EARLY_VARIABILITY_STD + 0.4)
    # Concrete anchor (Hướng B): name WHEN (the opening days) and WHAT scope
    # (day-to-day activity) so this doesn't paraphrase to a generic
    # "schedule" or "routine" overlap with consistent_routine.
    return InsightCandidate(
        pattern_family="X",
        pattern_type="early_variability",
        tentative_category="Opportunity",
        phase_focus="1",
        raw_template_key="early_variability",
        raw_signal={
            "subject": "your day-to-day activity in the opening days",
            "scope": "early_month_overall",
            "variability_label": _label_imbalance_std(std),
            "improvement_direction": "settling into a steadier opening pattern",
            "phase1_overall_std": round(std, 2),
        },
        signal_strength=round(strength, 3),
        cross_module_impact=0.50,
    )


def _increasing_variability(
    snapshot: MonthlySnapshot, phases: List[PhaseMetrics]
) -> Optional[InsightCandidate]:
    """Rules §1 Phase 2 row 5: 'Increasing variability — Variance ↑ vs Phase 1' → Need Attention."""
    s1 = phases[0].overall_day_score_std
    s2 = phases[1].overall_day_score_std
    if s1 is None or s2 is None or s1 == 0:
        return None
    growth = (s2 - s1) / s1
    if growth < _INCREASING_VARIANCE_PCT:
        return None
    strength = min(1.0, growth / 1.0)
    return InsightCandidate(
        pattern_family="X",
        pattern_type="increasing_variability",
        tentative_category="Need Attention",
        phase_focus="2",
        raw_template_key="increasing_variability",
        raw_signal={
            "subject": "your day-to-day activity consistency",
            "scope": "phase1_to_phase2_variability_growth",
            "direction_unambiguous": "swung more from day to day",
            "time_window_human": "by mid-month",
            "comparison_anchor": "compared to the opening weeks",
            "improvement_direction": "settling back into a steadier day-to-day rhythm",
            "phase1_std": round(s1, 2),
            "phase2_std": round(s2, 2),
            "growth": round(growth, 3),
        },
        signal_strength=round(strength, 3),
        cross_module_impact=0.50,
    )


def _cross_module_stability(
    snapshot: MonthlySnapshot, phases: List[PhaseMetrics]
) -> Optional[InsightCandidate]:
    """Rules §1 Phase 2 row 11: 'All stable — Low variance + no strong correlation' → Great Job."""
    s2 = phases[1].overall_day_score_std
    if s2 is None or s2 > _CROSS_STABLE_STD_CEIL:
        return None
    corrs = [
        snapshot.corr_work_health,
        snapshot.corr_work_spend,
        snapshot.corr_sleep_mood,
        snapshot.corr_workout_mood,
    ]
    corrs_ok = [c for c in corrs if c is not None]
    if not corrs_ok:
        return None
    if any(abs(c) > _CROSS_STABLE_CORR_CEIL for c in corrs_ok):
        return None
    strength = min(1.0, (_CROSS_STABLE_STD_CEIL - s2) / _CROSS_STABLE_STD_CEIL + 0.5)
    # Positive framing: the three areas stayed steady AND ran independently
    # (low |corr| = no negative drag) — "each kept its own pace, none dragged
    # the others". Distinct from stable_rhythm (cross-phase) and stable_rhythm_p2
    # (cross-module spread).
    return InsightCandidate(
        pattern_family="X",
        pattern_type="cross_module_stability",
        tentative_category="Great Job",
        phase_focus="2",
        raw_template_key="cross_module_stability",
        raw_signal={
            "subject": "your work, health, and spending",
            "scope": "mid_month_each_area_steady",
            "stability_label": "each_steady",
            "phase2_std": round(s2, 2),
            "max_abs_corr": round(max(abs(c) for c in corrs_ok), 3),
        },
        signal_strength=round(strength, 3),
        cross_module_impact=1.0,
    )


def _controlled_increase(
    snapshot: MonthlySnapshot, phases: List[PhaseMetrics]
) -> Optional[InsightCandidate]:
    """Rules §1 Phase 3 row 4: 'Work ↑, others stable' → Great Job.

    "Work ↑" is interpreted as Phase 3 vs Phase 2 (i.e. work picked up in
    the latest phase). This is the most actionable reading because §5
    "Conflict priority" lists trend before level, and a P3-vs-P2 trend is
    the freshest data point. P3-vs-P1 would also fit the wording but would
    weaken sensitivity to mid-month changes.
    """
    p2_work = phases[1].p_meeting_minutes_avg
    p3_work = phases[2].p_meeting_minutes_avg
    work_delta = pct_delta(p3_work, p2_work)
    if work_delta is None or work_delta < _CONTROLLED_INCREASE_PCT:
        return None
    others_stable = True
    for p2_v, p3_v in (
        (phases[1].h_health_score_avg, phases[2].h_health_score_avg),
        (phases[1].m_mood_score_avg, phases[2].m_mood_score_avg),
        (phases[1].financial_health_score_avg, phases[2].financial_health_score_avg),
    ):
        d = pct_delta(p3_v, p2_v)
        if d is None:
            continue
        if abs(d) > _CONTROLLED_INCREASE_STABLE_PCT:
            others_stable = False
            break
    if not others_stable:
        return None
    strength = min(1.0, work_delta / 0.50 + 0.4)
    return InsightCandidate(
        pattern_family="X",
        pattern_type="controlled_increase",
        tentative_category="Great Job",
        phase_focus="3",
        raw_template_key="controlled_increase",
        raw_signal={
            "subject": "your work output",
            "scope": "late_month_work_uptick",
            "direction_unambiguous": "picked back up",
            "time_window_human": "in the closing stretch of the month",
            "comparison_anchor": "compared to mid-month",
            "work_delta": round(work_delta, 3),
        },
        signal_strength=round(strength, 3),
        cross_module_impact=0.75,
    )


def _stress_pattern(
    snapshot: MonthlySnapshot, phases: List[PhaseMetrics]
) -> Optional[InsightCandidate]:
    """Rules §1 Phase 3 row 5: 'Work ↑ + Sleep ↓ + Variance ↑' → Need Attention."""
    p2, p3 = phases[1], phases[2]
    work_d = pct_delta(p3.p_meeting_minutes_avg, p2.p_meeting_minutes_avg)
    sleep_d = pct_delta(p3.h_sleep_hours_avg, p2.h_sleep_hours_avg)
    if work_d is None or sleep_d is None:
        return None
    if work_d < _STRESS_PATTERN_WORK_UP or sleep_d > _STRESS_PATTERN_SLEEP_DOWN:
        return None
    s2 = p2.overall_day_score_std or 0.0
    s3 = p3.overall_day_score_std or 0.0
    if s2 == 0 or (s3 - s2) / s2 < _STRESS_PATTERN_VAR_GROW:
        return None
    strength = min(1.0, (work_d + abs(sleep_d)) / 0.60)
    return InsightCandidate(
        pattern_family="X",
        pattern_type="stress_pattern",
        tentative_category="Need Attention",
        phase_focus="3",
        raw_template_key="stress_pattern",
        raw_signal={
            "subject": "your workload, sleep, and day-to-day swings together",
            "scope": "late_month_stress_signature",
            "direction_unambiguous": "work climbed while sleep slipped and swings grew",
            "time_window_human": "in the closing stretch",
            "comparison_anchor": "compared to mid-month",
            "improvement_direction": "easing workload and protecting sleep to settle the rhythm",
            "work_delta": round(work_d, 3),
            "sleep_delta": round(sleep_d, 3),
            "variance_growth": round((s3 - s2) / s2, 3),
        },
        signal_strength=round(strength, 3),
        cross_module_impact=1.0,
    )


def _mood_volatility(
    snapshot: MonthlySnapshot, phases: List[PhaseMetrics]
) -> Optional[InsightCandidate]:
    """Rules §3.B 'High variation' applied to mood — uses NEW DDL field
    ``m_mood_score_min`` / ``m_mood_score_max`` (daily intraday range).

    Phase-month mean of (max - min) feeds a continuous strength curve. The
    historical threshold ``_MOOD_VOLATILITY_RANGE`` (1.5) is preserved as
    the comment-anchor — it's the point where strength reaches ~0.33 — but
    no longer a hard cliff. Patterns near the anchor still surface with
    proportionally lower strength so ranking, not a binary cutoff, decides
    whether the user sees them (Feedback §5).
    """
    range_avg = snapshot.m_mood_range_avg
    if range_avg is None:
        return None
    # Linear ramp from range_avg=0.5 to range_avg=3.5, clamped to [0, 1].
    strength = max(0.0, min(1.0, (range_avg - 0.5) / 3.0))
    if strength < _GRADIENT_NOISE_FLOOR:
        return None
    return InsightCandidate(
        pattern_family="X",
        pattern_type="mood_volatility",
        tentative_category="Opportunity",
        phase_focus="all",
        raw_template_key="mood_volatility",
        raw_signal={
            "subject": "your mood swings within each day",
            "scope": "month_long_mood_intraday_range",
            "time_window_human": "across the month",
            "improvement_direction": "settling into steadier mood across each day",
            "volatility_label": _label_mood_range(range_avg),
            "mood_range_avg": round(range_avg, 2),
        },
        signal_strength=round(strength, 3),
        cross_module_impact=0.50,
    )


# =============================================================================
# Historical detectors — Rules §1 Phase 1/2/3 cross-month logic
# Fire only when ``BaselineContext`` carries the relevant past data. All
# detectors guard for missing fields and return ``None`` gracefully.
# =============================================================================


def _baseline_metric_avg(snapshots: List[MonthlySnapshot], key: str) -> Optional[float]:
    """Average of ``key`` across snapshots (None-safe)."""
    vals = [getattr(s, key, None) for s in snapshots]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 3)


def _phase1_module_avgs(phase: PhaseMetrics) -> Dict[str, float]:
    """Per-module Phase 1 averages on a 0-100 scale.

    Aligned with ``balance_score`` (mean of h_health_score, productivity_score,
    financial_health_score). Mood is excluded — Rules §1 P1 rows 1-2 say
    "ALL MODULES ≥/< baseline" without nominating mood, and the rest of the
    balance plumbing intentionally omits mood for the same reason.
    """
    out: Dict[str, float] = {}
    if phase.h_health_score_avg is not None:
        out["h_health_score_avg"] = float(phase.h_health_score_avg)
    if phase.productivity_score_avg is not None:
        out["productivity_score_avg"] = float(phase.productivity_score_avg)
    if phase.financial_health_score_avg is not None:
        out["financial_health_score_avg"] = float(phase.financial_health_score_avg)
    return out


def _baseline_module_value(
    snap: Optional[MonthlySnapshot], avg_dict: Dict[str, Optional[float]], key: str
) -> Optional[float]:
    """Resolve module baseline value from prev-month snapshot or an avg dict.

    All 3 balance modules (h_health, productivity, financial_health) are stored
    on 0-100 scale, so no normalization is required.
    """
    if snap is not None:
        val = getattr(snap, key, None)
        if val is not None:
            return float(val)
    val = avg_dict.get(key)
    if val is None:
        return None
    return float(val)


def _higher_than_usual_start(
    snapshot: MonthlySnapshot,
    phases: List[PhaseMetrics],
    baseline: BaselineContext,
) -> Optional[InsightCandidate]:
    """Rules §1 Phase 1 row 1: 'All modules ≥ prev month avg OR ≥90d avg' → Great Job.

    Spec demands the check be per-module ("ALL modules") AND treats prev-month /
    90d as alternative baselines (OR). For each available module we require its
    Phase-1 value to clear at least one baseline by ``_LEVEL_VS_HISTORY_PCT``.
    """
    p1 = phases[0]
    if phases[0].overall_day_score_avg is None or p1.days_with_data < 3:
        return None
    if snapshot.days_with_data > _EARLY_DAYS_CAP:
        return None
    p1_mods = _phase1_module_avgs(p1)
    if len(p1_mods) < _MIN_MODULES_FOR_BALANCE:
        return None
    prev = baseline.prev_month_snapshot
    avg90 = baseline.avg_90d
    has_any_baseline = False
    for key, cur in p1_mods.items():
        prev_val = _baseline_module_value(prev, {}, key) if prev else None
        avg_val = _baseline_module_value(None, avg90, key)
        refs = [r for r in (prev_val, avg_val) if r is not None]
        if not refs:
            return None  # cannot verify this module → fail "ALL modules"
        has_any_baseline = True
        if not any(cur >= r * (1.0 + _LEVEL_VS_HISTORY_PCT) for r in refs):
            return None
    if not has_any_baseline:
        return None
    overall_p1 = phases[0].overall_day_score_avg or 0.0
    overall_avg90 = avg90.get("overall_day_score_avg")
    overall_prev = prev.overall_day_score_avg if prev else None
    refs_overall = [r for r in (overall_prev, overall_avg90) if r is not None]
    delta = (pct_delta(overall_p1, max(refs_overall)) if refs_overall else None) or 0.0
    strength = min(1.0, abs(delta) / 0.50 + 0.4)
    return InsightCandidate(
        pattern_family="X",
        pattern_type="higher_than_usual_start",
        tentative_category="Great Job",
        phase_focus="1",
        raw_template_key="higher_than_usual_start",
        raw_signal={
            "subject": "your day-to-day activity across health, work, and finance",
            "scope": "early_month_above_history",
            "direction_unambiguous": "started above the usual level",
            "time_window_human": "early in the month",
            "comparison_anchor": "compared to last month and the 90-day average",
            "phase1_avg": round(overall_p1, 2),
            "prev_month_avg": (
                round(overall_prev, 2) if overall_prev is not None else None
            ),
            "avg_90d": overall_avg90,
            "delta": round(delta, 3),
        },
        signal_strength=round(strength, 3),
        cross_module_impact=0.50,
    )


def _lower_than_usual_start(
    snapshot: MonthlySnapshot,
    phases: List[PhaseMetrics],
    baseline: BaselineContext,
) -> Optional[InsightCandidate]:
    """Rules §1 Phase 1 row 2: 'All modules < prev month avg AND <90d avg' → Opportunity.

    Spec uses "AND" between the conditions (prev month + 90d) — at the
    baseline-availability level we accept a single baseline (graceful when
    90d isn't established), but for users with both we require each module
    to be below every available baseline by ``_LEVEL_VS_HISTORY_PCT``.
    """
    p1 = phases[0]
    if phases[0].overall_day_score_avg is None or p1.days_with_data < 3:
        return None
    if snapshot.days_with_data > _EARLY_DAYS_CAP:
        return None
    p1_mods = _phase1_module_avgs(p1)
    if len(p1_mods) < _MIN_MODULES_FOR_BALANCE:
        return None
    prev = baseline.prev_month_snapshot
    avg90 = baseline.avg_90d
    has_any_baseline = False
    for key, cur in p1_mods.items():
        prev_val = _baseline_module_value(prev, {}, key) if prev else None
        avg_val = _baseline_module_value(None, avg90, key)
        refs = [r for r in (prev_val, avg_val) if r is not None]
        if not refs:
            return None  # cannot verify this module → fail "ALL modules"
        has_any_baseline = True
        if not all(cur < r * (1.0 - _LEVEL_VS_HISTORY_PCT) for r in refs):
            return None
    if not has_any_baseline:
        return None
    overall_p1 = phases[0].overall_day_score_avg or 0.0
    overall_avg90 = avg90.get("overall_day_score_avg")
    overall_prev = prev.overall_day_score_avg if prev else None
    refs_overall = [r for r in (overall_prev, overall_avg90) if r is not None]
    delta = (pct_delta(overall_p1, min(refs_overall)) if refs_overall else None) or 0.0
    strength = min(1.0, abs(delta) / 0.50 + 0.3)
    return InsightCandidate(
        pattern_family="X",
        pattern_type="lower_than_usual_start",
        tentative_category="Opportunity",
        phase_focus="1",
        raw_template_key="lower_than_usual_start",
        raw_signal={
            "subject": "your day-to-day activity across health, work, and finance",
            "scope": "early_month_below_history",
            "direction_unambiguous": "started below the usual level",
            "time_window_human": "early in the month",
            "comparison_anchor": "compared to last month and the 90-day average",
            "improvement_direction": "lifting the opening weeks back toward the usual level",
            "phase1_avg": round(overall_p1, 2),
            "prev_month_avg": (
                round(overall_prev, 2) if overall_prev is not None else None
            ),
            "avg_90d": overall_avg90,
            "delta": round(delta, 3),
        },
        signal_strength=round(strength, 3),
        cross_module_impact=0.50,
    )


def _change_vs_last_month_start(
    snapshot: MonthlySnapshot,
    phases: List[PhaseMetrics],
    baseline: BaselineContext,
) -> Optional[InsightCandidate]:
    """Rules §1 Phase 1 row 3: 'Current start ≠ last month start by >20%' → Opportunity.

    Skipped when ``_higher_than_usual_start`` / ``_lower_than_usual_start``
    would also fire — those carry stronger, more specific framing for the same
    20%+ deviation, so the generic "change vs last month" insight would only
    duplicate the slot.
    """
    p1 = phases[0].overall_day_score_avg
    if p1 is None or phases[0].days_with_data < 3:
        return None
    if snapshot.days_with_data > _EARLY_DAYS_CAP:
        return None
    if not baseline.prev_month_snapshot or not baseline.prev_month_snapshot.phases:
        return None
    prev_p1 = baseline.prev_month_snapshot.phases[0].overall_day_score_avg
    if prev_p1 is None:
        return None
    delta = pct_delta(p1, prev_p1)
    if delta is None or abs(delta) < _START_CHANGE_PCT:
        return None
    # Skip when higher/lower-than-usual would also fire on the same deviation.
    # Each helper here re-runs its per-module loop; cost is acceptable because
    # historical detectors only run once per snapshot generation (Step 4).
    if _higher_than_usual_start(snapshot, phases, baseline) is not None:
        return None
    if _lower_than_usual_start(snapshot, phases, baseline) is not None:
        return None
    strength = min(1.0, abs(delta) / 0.40 + 0.3)
    return InsightCandidate(
        pattern_family="X",
        pattern_type="change_vs_last_month_start",
        tentative_category="Opportunity",
        phase_focus="1",
        raw_template_key="change_vs_last_month_start",
        raw_signal={
            "subject": "your day-to-day activity at the start of the month",
            "scope": "early_month_change_vs_last_month",
            "direction_unambiguous": "shifted from last month's start",
            "time_window_human": "early in the month",
            "comparison_anchor": "compared to the start of last month",
            "phase1_avg": round(p1, 2),
            "prev_month_phase1_avg": round(prev_p1, 2),
            "delta": round(delta, 3),
        },
        signal_strength=round(strength, 3),
        cross_module_impact=0.40,
    )


def _flat_start_vs_90d(
    snapshot: MonthlySnapshot,
    phases: List[PhaseMetrics],
    baseline: BaselineContext,
) -> Optional[InsightCandidate]:
    """Rules §1 Phase 1 row 8: 'All modules low vs 90d baseline' → Opportunity.

    Distinguished from ``_early_low_activity`` which uses an absolute cutoff
    (``_EARLY_LOW_SCORE``); this uses baseline-relative comparison.
    """
    p1 = phases[0].overall_day_score_avg
    if p1 is None or phases[0].days_with_data < 3:
        return None
    if snapshot.days_with_data > _EARLY_DAYS_CAP:
        return None
    avg90 = baseline.avg_90d.get("overall_day_score_avg")
    if avg90 is None or avg90 <= 0:
        return None
    delta = pct_delta(p1, avg90)
    if delta is None or delta > -_LEVEL_VS_HISTORY_PCT:
        return None
    strength = min(1.0, abs(delta) / 0.50 + 0.3)
    return InsightCandidate(
        pattern_family="X",
        pattern_type="flat_start_vs_90d",
        tentative_category="Opportunity",
        phase_focus="1",
        raw_template_key="flat_start_vs_90d",
        raw_signal={
            "subject": "your day-to-day activity in the opening weeks",
            "scope": "early_month_flat_vs_90d_baseline",
            "direction_unambiguous": "trailed the 90-day baseline",
            "time_window_human": "early in the month",
            "comparison_anchor": "compared to the 90-day average",
            "improvement_direction": "nudging activity up across the board to catch up to your usual baseline",
            "phase1_avg": round(p1, 2),
            "avg_90d": avg90,
            "delta": round(delta, 3),
        },
        signal_strength=round(strength, 3),
        cross_module_impact=0.50,
    )


def _positive_carry_over(
    snapshot: MonthlySnapshot,
    phases: List[PhaseMetrics],
    baseline: BaselineContext,
) -> Optional[InsightCandidate]:
    """Rules §1 Phase 1 row 9: 'Same strong pattern as last month' → Great Job.

    Requirements:
      1. Last month had a ``Great Job`` insight in ``_POSITIVE_CARRY_OVER_TYPES``
         with score ≥ ``_CARRY_OVER_MIN_SIGNAL``.
      2. This month's Phase 1 metric corresponding to that pattern still holds
         (≥ -10% drift from prev month overall avg AND the pattern-specific
         metric remains within its trigger band).
    Exempt from novelty penalty (see ``_CARRY_OVER_EXEMPT_TYPES``).
    """
    if not baseline.prev_month_insights:
        return None
    eligible = [
        i
        for i in baseline.prev_month_insights
        if i.category == "Great Job"
        and i.score >= _CARRY_OVER_MIN_SIGNAL
        and i.pattern_type in _POSITIVE_CARRY_OVER_TYPES
    ]
    if not eligible:
        return None
    p1 = phases[0].overall_day_score_avg
    prev_avg = (
        baseline.prev_month_snapshot.overall_day_score_avg
        if baseline.prev_month_snapshot
        else None
    )
    if p1 is None or prev_avg is None:
        return None
    delta = pct_delta(p1, prev_avg) or 0.0
    if delta < -_HABIT_180D_MATCH_PCT:
        return None  # ≥10% drop ⇒ no longer the "same strong pattern"
    top = max(eligible, key=lambda i: i.score)
    # Verify the SAME pattern's underlying metric still holds this month.
    if not _carry_over_pattern_holds(top.pattern_type, snapshot, phases):
        return None
    strength = min(1.0, top.score + 0.2)
    return InsightCandidate(
        pattern_family="X",
        pattern_type="positive_carry_over",
        tentative_category="Great Job",
        phase_focus="1",
        raw_template_key="positive_carry_over",
        raw_signal={
            "subject": "the strong pattern carried over from last month",
            "scope": "early_month_positive_carry_over",
            "direction_unambiguous": "kept holding",
            "time_window_human": "early in the month",
            "comparison_anchor": "compared to last month",
            "prev_month_pattern_type": top.pattern_type,
            "prev_month_score": round(top.score, 3),
            "phase1_avg": round(p1, 2),
            "prev_month_avg": round(prev_avg, 2),
        },
        signal_strength=round(strength, 3),
        cross_module_impact=0.50,
    )


def _negative_carry_over(
    snapshot: MonthlySnapshot,
    phases: List[PhaseMetrics],
    baseline: BaselineContext,
) -> Optional[InsightCandidate]:
    """Rules §1 Phase 1 row 10: 'Same imbalance as last month' → Need Attention (low confidence).

    Requirements:
      1. Last month had a Need-Attention insight in
         ``_NEGATIVE_CARRY_OVER_TYPES`` with score ≥ ``_CARRY_OVER_MIN_SIGNAL``.
      2. This month's imbalance (stddev of 3 modules) is again ≥
         ``_IMBALANCE_HIGH_STD`` — same imbalance recurring. With the
         mean-based balance_score, level alone wouldn't catch "same
         imbalance"; spread (std) is the spec-correct signal.
      3. The dominant module (``most_concentrated_area``) matches last
         month's — required by spec wording "SAME imbalance" (not just any
         imbalance). When the previous Insight has no recorded area we fall
         back to imbalance-only check.
    Strength is tiered by calendar day so early-month firings get suppressed
    by ``_drop_early_weak`` while late-month firings compete fairly.
    """
    if not baseline.prev_month_insights:
        return None
    eligible = [
        i
        for i in baseline.prev_month_insights
        if i.category == "Need Attention"
        and i.score >= _CARRY_OVER_MIN_SIGNAL
        and i.pattern_type in _NEGATIVE_CARRY_OVER_TYPES
    ]
    if not eligible:
        return None
    imbalance_std = _monthly_imbalance_std(snapshot)
    if imbalance_std is None or imbalance_std < _IMBALANCE_HIGH_STD:
        return None
    # Same-module check: prev top NA insight must point at the same module
    # that is dominant this month.
    cur_area = snapshot.most_concentrated_area
    same_area_matches = [
        i
        for i in eligible
        if cur_area is not None
        and isinstance(i.raw_signal, dict)
        and i.raw_signal.get("area") == cur_area
    ]
    if same_area_matches:
        top = max(same_area_matches, key=lambda i: i.score)
    else:
        # Fallback: when no prev insight recorded an area (older history),
        # accept the strongest NA-imbalance signal — preserves recall while
        # logging the gap for diagnostic visibility.
        top = max(eligible, key=lambda i: i.score)
    today = date.today()
    in_early_window = (
        today.year == snapshot.year
        and today.month == snapshot.month
        and today.day <= _EARLY_CALENDAR_DAY_LIMIT
    )
    if in_early_window:
        # Early window: keep strength < _EXTREME_STRENGTH so _drop_early_weak
        # can suppress per Rules §5 row 1.
        strength = min(0.55, top.score)
    else:
        # Late window or past months: allow carry-over to compete in top-N.
        strength = min(0.85, top.score + 0.2)
    return InsightCandidate(
        pattern_family="X",
        pattern_type="negative_carry_over",
        tentative_category="Need Attention",
        phase_focus="1",
        raw_template_key="negative_carry_over",
        raw_signal={
            "subject": "the imbalance carried over from last month",
            "scope": "early_month_negative_carry_over",
            "direction_unambiguous": "kept recurring",
            "time_window_human": "early in the month",
            "comparison_anchor": "compared to last month",
            "improvement_direction": "breaking the recurring imbalance pattern",
            "prev_month_pattern_type": top.pattern_type,
            "prev_month_score": round(top.score, 3),
            "confidence": "low",
        },
        signal_strength=round(strength, 3),
        cross_module_impact=0.50,
    )


def _carry_over_pattern_holds(
    pattern_type: str,
    snapshot: MonthlySnapshot,
    phases: List[PhaseMetrics],
) -> bool:
    """Return True if the named positive pattern still has supporting metrics
    in the current month (Rules §1 P1 row 9 "same STRONG pattern as last month").

    Lightweight surrogate check — covers every type in
    ``_POSITIVE_CARRY_OVER_TYPES``. Unknown pattern types fail the check (no
    silent pass-through) so we don't claim "same strong pattern" without
    evidence.
    """
    # Stability / balance-family patterns — verify the user is still
    # "balanced" this month. With mean-based balance_score, "balanced" means
    # modules are CLOSE together (low spread), so we check stddev of the 3
    # modules against the imbalance threshold. Missing data ⇒ trust prev.
    if pattern_type in {
        "stable_rhythm",
        "balanced_start",
        "sustained_balance",
        "cross_module_stability",
    }:
        imbalance_std = _monthly_imbalance_std(snapshot)
        return imbalance_std is None or imbalance_std < _IMBALANCE_HIGH_STD
    # Routine consistency — require non-noisy sleep std this month
    if pattern_type == "consistent_routine":
        worst = max(
            (p.h_sleep_hours_std for p in phases if p.h_sleep_hours_std is not None),
            default=None,
        )
        return worst is None or worst < _SLEEP_HOURS_NOISY_STD
    # Focus / social / spending — simple metric checks
    if pattern_type == "focused_work_blocks":
        return (snapshot.p_meeting_minutes_avg or 0.0) <= _FEWER_MEETINGS_CEILING
    if pattern_type == "social_energy_days":
        return (snapshot.corr_meeting_mood or 0.0) >= 0.0
    if pattern_type == "controlled_spending":
        c = snapshot.corr_work_spend
        return c is None or abs(c) <= _CORR_CONTROLLED_CEILING * 1.5
    if pattern_type == "workout_mood_chain":
        return (snapshot.corr_workout_mood or 0.0) >= 0.0
    # Goal alignment is re-checked by its own detector each month;
    # 180d habits self-verify via Pearson/avg comparison. Trust prev signal.
    if pattern_type in {
        "consistent_goal_alignment",
        "habit_alignment_180d",
        "habit_180d",
    }:
        return True
    # Unknown type — fail closed (Rules §1 P1 row 9 must have current-month
    # evidence; we should not claim continuation without it).
    return False


def _early_tradeoff_vs_baseline(
    snapshot: MonthlySnapshot,
    phases: List[PhaseMetrics],
    baseline: BaselineContext,
) -> Optional[InsightCandidate]:
    """Rules §1 Phase 1 row 7: 'Work ↑ + Health ↓ vs prev month baseline' → Need Attention (low confidence).

    Distinct from ``_work_health_tradeoff_corr`` which uses month-wide
    correlation. This is the Phase-1 EARLY signal: comparing this month's
    Phase 1 work load and health score against last month's full-month
    averages. Capped at ``signal_strength < 0.60`` (extreme) so
    ``_drop_early_weak`` still suppresses it in days 1-5, and explicitly
    marked ``confidence=low`` per spec parenthetical.
    """
    if not baseline.prev_month_snapshot:
        return None
    if snapshot.days_with_data > _EARLY_DAYS_CAP:
        return None
    p1 = phases[0]
    if p1.days_with_data < 3:
        return None

    cur_work = p1.p_meeting_minutes_avg
    cur_health = p1.h_health_score_avg
    prev_work = baseline.prev_month_snapshot.p_meeting_minutes_avg
    prev_health = baseline.prev_month_snapshot.h_health_score_avg
    if (
        cur_work is None
        or cur_health is None
        or prev_work is None
        or prev_health is None
    ):
        return None

    work_delta = pct_delta(cur_work, prev_work)
    health_delta = pct_delta(cur_health, prev_health)
    if work_delta is None or health_delta is None:
        return None
    # Work must rise meaningfully AND health must drop meaningfully.
    # Re-use _WORK_RISE_PCT (0.20) and a mirrored drop threshold.
    if work_delta < _WORK_RISE_PCT:
        return None
    if health_delta > -_LEVEL_VS_HISTORY_PCT:  # health drop ≥ 20%
        return None

    # Strength scaled by combined magnitude; capped below "extreme" (0.60)
    # so _drop_early_weak suppresses unless both deltas are large.
    combined = min(1.0, (work_delta + abs(health_delta)) / 0.80)
    strength = min(0.55, 0.30 + combined * 0.30)
    return InsightCandidate(
        pattern_family="X",
        pattern_type="early_tradeoff_vs_baseline",
        tentative_category="Need Attention",
        phase_focus="1",
        raw_template_key="early_tradeoff_vs_baseline",
        raw_signal={
            "subject": "your workload and health together in the opening weeks",
            "scope": "early_month_work_up_health_down_vs_baseline",
            "direction_unambiguous": "workload rose while health dropped",
            "time_window_human": "early in the month",
            "comparison_anchor": "compared to last month",
            "improvement_direction": "protecting health time as workload picks up",
            "phase1_work_avg": round(cur_work, 2),
            "prev_month_work_avg": round(prev_work, 2),
            "work_delta": round(work_delta, 3),
            "phase1_health_avg": round(cur_health, 2),
            "prev_month_health_avg": round(prev_health, 2),
            "health_delta": round(health_delta, 3),
            "confidence": "low",
        },
        signal_strength=round(strength, 3),
        cross_module_impact=0.75,
    )


def _habit_alignment_180d(
    snapshot: MonthlySnapshot,
    phases: List[PhaseMetrics],
    baseline: BaselineContext,
) -> Optional[InsightCandidate]:
    """Rules §1 Phase 1 row 11 + Phase 3 row 13: 'Behaviour aligns with 180d trend' → Great Job."""
    if not baseline.has_180d_baseline():
        return None
    p1 = phases[0].overall_day_score_avg
    avg180 = baseline.avg_180d.get("overall_day_score_avg")
    if p1 is None or avg180 is None or avg180 <= 0:
        return None
    delta = pct_delta(p1, avg180)
    if delta is None or abs(delta) > _HABIT_180D_MATCH_PCT:
        return None
    # phase_focus "cross" — applies across early month AND month-end synthesis
    strength = 0.5 + (_HABIT_180D_MATCH_PCT - abs(delta)) * 2.0
    return InsightCandidate(
        pattern_family="X",
        pattern_type="habit_alignment_180d",
        tentative_category="Great Job",
        phase_focus="cross",
        raw_template_key="habit_alignment_180d",
        raw_signal={
            "subject": "your day-to-day activity against the long-term habit",
            "scope": "early_month_aligned_with_180d_habit",
            "direction_unambiguous": "lined up with the long-term habit",
            "time_window_human": "early in the month",
            "comparison_anchor": "compared to the 180-day average",
            "phase1_avg": round(p1, 2),
            "avg_180d": round(avg180, 2),
            "delta": round(delta, 3),
        },
        signal_strength=round(min(1.0, strength), 3),
        cross_module_impact=0.60,
    )


def _divergence_vs_history_p2(
    snapshot: MonthlySnapshot,
    phases: List[PhaseMetrics],
    baseline: BaselineContext,
) -> Optional[InsightCandidate]:
    """Rules §1 Phase 2 row 3: 'Phase 2 differs from previous month by >20%' → Opportunity."""
    p2 = phases[1].overall_day_score_avg
    if p2 is None or phases[1].days_with_data < 3:
        return None
    if not baseline.prev_month_snapshot or not baseline.prev_month_snapshot.phases:
        return None
    prev_p2 = baseline.prev_month_snapshot.phases[1].overall_day_score_avg
    if prev_p2 is None:
        return None
    delta = pct_delta(p2, prev_p2)
    if delta is None or abs(delta) < _DIVERGENCE_VS_HISTORY_PCT:
        return None
    strength = min(1.0, abs(delta) / 0.40 + 0.3)
    return InsightCandidate(
        pattern_family="X",
        pattern_type="divergence_vs_history_p2",
        tentative_category="Opportunity",
        phase_focus="2",
        raw_template_key="divergence_vs_history_p2",
        raw_signal={
            "subject": "your mid-month day-to-day activity",
            "scope": "mid_month_diverging_from_history",
            "direction_unambiguous": "drifted from last month's mid-month",
            "time_window_human": "mid-month",
            "comparison_anchor": "compared to last month's mid-month",
            "phase2_avg": round(p2, 2),
            "prev_month_phase2_avg": round(prev_p2, 2),
            "delta": round(delta, 3),
        },
        signal_strength=round(strength, 3),
        cross_module_impact=0.40,
    )


def _pattern_repeat_90d(
    snapshot: MonthlySnapshot,
    phases: List[PhaseMetrics],
    baseline: BaselineContext,
) -> Optional[InsightCandidate]:
    """Rules §1 Phase 2 row 12: 'Same pattern in ≥2 of last 3 months'.

    Category mirrors the recurring pattern's category (Great Job if recurring
    pattern is positive, Need Attention if negative). Exempt from novelty
    penalty so the recurrence IS the signal.

    Also verifies the CURRENT month still exhibits the recurring pattern —
    using ``_pattern_holds_in_current_month`` — so we don't tell a user
    "this pattern keeps coming back" when in fact it has cleared this month.
    """
    if not baseline.past_insights_by_month:
        return None
    # Only consider last 3 months
    sorted_months = sorted(baseline.past_insights_by_month.keys(), reverse=True)[:3]
    if len(sorted_months) < 2:
        return None
    # Count pattern_type occurrences across last 3 months
    pattern_counts: Dict[str, int] = {}
    pattern_categories: Dict[str, str] = {}
    for ym in sorted_months:
        seen_types_this_month: set = set()
        for ins in baseline.past_insights_by_month[ym]:
            if ins.pattern_type in seen_types_this_month:
                continue
            seen_types_this_month.add(ins.pattern_type)
            pattern_counts[ins.pattern_type] = (
                pattern_counts.get(ins.pattern_type, 0) + 1
            )
            pattern_categories[ins.pattern_type] = ins.category
    recurring = [
        p for p, c in pattern_counts.items() if c >= _PATTERN_REPEAT_MIN_OCCURRENCES
    ]
    if not recurring:
        return None
    # Pick the strongest recurring pattern that ALSO holds this month.
    recurring.sort(key=lambda p: (-pattern_counts[p], p))
    chosen: Optional[str] = None
    for ptype in recurring:
        if _pattern_holds_in_current_month(ptype, snapshot, phases):
            chosen = ptype
            break
    if chosen is None:
        return None
    top_cat = pattern_categories[chosen]
    if top_cat not in ("Great Job", "Need Attention", "Opportunity"):
        return None
    strength = min(1.0, pattern_counts[chosen] / 3.0 + 0.4)
    return InsightCandidate(
        pattern_family="X",
        pattern_type="pattern_repeat_90d",
        tentative_category=top_cat,  # type: ignore[arg-type]
        phase_focus="2",
        raw_template_key="pattern_repeat_90d",
        raw_signal={
            "subject": "a pattern that keeps repeating across recent months",
            "scope": "mid_month_pattern_repeat_90d",
            "direction_unambiguous": "kept showing up across recent months",
            "time_window_human": "mid-month",
            "comparison_anchor": "compared to the last three months",
            "recurring_pattern_type": chosen,
            "occurrences_in_last_3_months": pattern_counts[chosen],
            "carried_category": top_cat,
        },
        signal_strength=round(strength, 3),
        cross_module_impact=0.60,
    )


def _pattern_holds_in_current_month(
    pattern_type: str,
    snapshot: MonthlySnapshot,
    phases: List[PhaseMetrics],
) -> bool:
    """Lightweight verification that the named pattern still has supporting
    metrics in the current month. Used by ``_pattern_repeat_90d`` and
    ``_reinforcement_90d`` to avoid surfacing "this keeps recurring" when the
    pattern has actually cleared.

    Rules §1 P2 row 12 / P3 row 12 require "same pattern" / "same outcome" —
    i.e. the pattern MUST be present this month for the recurrence claim to
    hold. Unknown pattern types fail closed (return False) so we don't claim
    recurrence without verification.
    """
    if pattern_type == "cross_module_imbalance":
        # Mean-based balance_score can't detect imbalance — use spread (stddev
        # of the 3 modules) to mirror ``_cross_module_imbalance`` detector.
        imbalance_std = _monthly_imbalance_std(snapshot)
        return imbalance_std is not None and imbalance_std >= _IMBALANCE_HIGH_STD
    if pattern_type == "skewed_allocation":
        return (snapshot.module_max_excess_pct or 0.0) >= _SKEW_EXCESS_PCT
    if pattern_type == "meeting_overload":
        # Mirrors ``_meeting_overload`` — day-count first, monthly average as
        # backwards-compat fallback for snapshots predating ``p_heavy_meeting_days``.
        heavy_days = sum((p.p_heavy_meeting_days or 0) for p in phases)
        if heavy_days >= _MEETING_HEAVY_MIN_DAYS:
            return True
        return (
            snapshot.p_meeting_minutes_avg or 0.0
        ) >= _MEETING_HEAVY_DAY_THRESHOLD_MIN
    if pattern_type == "stress_linked_spending":
        return (snapshot.corr_work_spend or 0.0) >= _CORR_STRESS_SPEND
    if pattern_type == "work_health_tradeoff_corr":
        c = snapshot.corr_work_health
        return c is not None and c <= _NEG_CORR_THRESHOLD
    if pattern_type == "sleep_work_chain":
        c = snapshot.corr_work_sleep
        return c is not None and c <= _NEG_CORR_THRESHOLD
    if pattern_type == "late_work_impact":
        return (snapshot.h_sleep_hours_avg or 8.0) < 7.0
    if pattern_type == "controlled_spending":
        c = snapshot.corr_work_spend
        return c is not None and abs(c) <= _CORR_CONTROLLED_CEILING
    if pattern_type == "workout_mood_chain":
        return (snapshot.corr_workout_mood or 0.0) >= _POS_CORR_THRESHOLD
    if pattern_type == "social_energy_days":
        return (snapshot.corr_meeting_mood or 0.0) >= _SOCIAL_ENERGY_CORR
    # Fail closed for unknown types — spec wording "same pattern" requires
    # verifiable current-month evidence.
    return False


def _behaviour_shift_180d(
    snapshot: MonthlySnapshot,
    phases: List[PhaseMetrics],
    baseline: BaselineContext,
) -> Optional[InsightCandidate]:
    """Rules §1 Phase 2 row 13: 'Pattern differs from long-term baseline' → Opportunity.

    ``_BEHAVIOUR_SHIFT_PCT`` (0.20) is preserved as a documentary anchor — at
    that |delta| the new continuous formula produces strength ≈ 0.40, roughly
    matching the original cliff-edge intensity. Smaller deltas now surface
    with proportionally lower strength rather than being silenced (Feedback §5).
    """
    if not baseline.has_180d_baseline():
        return None
    cur = snapshot.overall_day_score_avg
    avg180 = baseline.avg_180d.get("overall_day_score_avg")
    if cur is None or avg180 is None or avg180 <= 0:
        return None
    delta = pct_delta(cur, avg180)
    if delta is None:
        return None
    strength = max(0.0, min(1.0, abs(delta) / 0.50))
    if strength < _GRADIENT_NOISE_FLOOR:
        return None
    return InsightCandidate(
        pattern_family="X",
        pattern_type="behaviour_shift_180d",
        tentative_category="Opportunity",
        phase_focus="2",
        raw_template_key="behaviour_shift_180d",
        raw_signal={
            "subject": "your day-to-day activity against the long-term habit",
            "scope": "mid_month_behaviour_shift_vs_180d",
            "direction_unambiguous": "drifted from the long-term habit",
            "time_window_human": "mid-month",
            "comparison_anchor": "compared to the 180-day average",
            "current_month_avg": round(cur, 2),
            "avg_180d": round(avg180, 2),
            "delta": round(delta, 3),
        },
        signal_strength=round(strength, 3),
        cross_module_impact=0.40,
    )


def _improvement_vs_last_month(
    snapshot: MonthlySnapshot,
    phases: List[PhaseMetrics],
    baseline: BaselineContext,
) -> Optional[InsightCandidate]:
    """Rules §1 Phase 3 row 10: 'Better balance vs previous month' → Great Job.

    balance_score is now mean(3 modules) on 0..100 scale: HIGHER = better.
    Fires when current_balance > prev_month_balance by at least
    ``_MONTH_CHANGE_BAL_PCT`` points.
    """
    if not baseline.prev_month_snapshot:
        return None
    cur_bs = snapshot.balance_score
    prev_bs = baseline.prev_month_snapshot.balance_score
    if cur_bs is None or prev_bs is None:
        return None
    # Mean-based: cur > prev = improvement (level rising across modules).
    delta = cur_bs - prev_bs
    if delta < _MONTH_CHANGE_BAL_PCT:
        return None
    # Normalize over a 30-point band → 0..1 strength with 0.4 floor.
    strength = min(1.0, delta / 30.0 + 0.4)
    return InsightCandidate(
        pattern_family="X",
        pattern_type="improvement_vs_last_month",
        tentative_category="Great Job",
        phase_focus="3",
        raw_template_key="improvement_vs_last_month",
        raw_signal={
            "subject": "the balance across health, work, and finance compared to last month",
            "scope": "late_month_improvement_vs_last_month",
            "direction_unambiguous": "improved",
            "time_window_human": "in the closing stretch",
            "comparison_anchor": "compared to last month",
            "magnitude_label": _label_balance_drop(delta),
            "direction": "improved",
            "current_balance_score": round(cur_bs, 2),
            "prev_month_balance_score": round(prev_bs, 2),
            "improvement": round(delta, 2),
        },
        signal_strength=round(strength, 3),
        cross_module_impact=0.60,
    )


def _deterioration_vs_last_month(
    snapshot: MonthlySnapshot,
    phases: List[PhaseMetrics],
    baseline: BaselineContext,
) -> Optional[InsightCandidate]:
    """Rules §1 Phase 3 row 11: 'Worse trade-offs vs previous month' → Need Attention.

    Mirror of ``_improvement_vs_last_month`` — fires when cur < prev by at
    least ``_MONTH_CHANGE_BAL_PCT`` points (level dropping across modules).
    """
    if not baseline.prev_month_snapshot:
        return None
    cur_bs = snapshot.balance_score
    prev_bs = baseline.prev_month_snapshot.balance_score
    if cur_bs is None or prev_bs is None:
        return None
    delta = prev_bs - cur_bs  # positive ⇒ current lower ⇒ deterioration
    if delta < _MONTH_CHANGE_BAL_PCT:
        return None
    strength = min(1.0, delta / 30.0 + 0.4)
    return InsightCandidate(
        pattern_family="X",
        pattern_type="deterioration_vs_last_month",
        tentative_category="Need Attention",
        phase_focus="3",
        raw_template_key="deterioration_vs_last_month",
        raw_signal={
            "subject": "the balance across health, work, and finance compared to last month",
            "scope": "late_month_deterioration_vs_last_month",
            "direction_unambiguous": "declined",
            "time_window_human": "in the closing stretch",
            "comparison_anchor": "compared to last month",
            "improvement_direction": "rebuilding an even pace across health, work, and finance",
            "magnitude_label": _label_balance_drop(delta),
            "direction": "declined",
            "current_balance_score": round(cur_bs, 2),
            "prev_month_balance_score": round(prev_bs, 2),
            "deterioration": round(delta, 2),
        },
        signal_strength=round(strength, 3),
        cross_module_impact=0.60,
    )


def _reinforcement_90d(
    snapshot: MonthlySnapshot,
    phases: List[PhaseMetrics],
    baseline: BaselineContext,
) -> Optional[InsightCandidate]:
    """Rules §1 Phase 3 row 12: 'Same outcome across 3 months' → mirror category.

    Spec table cell reads "Great Job" but template text "This pattern is
    consistent with your recent months." is neutral observation. To stay
    consistent with ``_pattern_repeat_90d`` (P2.12) and avoid praising a
    persistent Need-Attention pattern as Great Job, we mirror the recurring
    pattern's category instead of restricting to Great Job only.
    """
    if not baseline.past_insights_by_month:
        return None
    sorted_months = sorted(baseline.past_insights_by_month.keys(), reverse=True)[:3]
    if len(sorted_months) < _REINFORCEMENT_PAST_MIN:
        return None
    type_counts: Dict[str, int] = {}
    type_categories: Dict[str, str] = {}
    for ym in sorted_months:
        seen_types_this_month: set = set()
        for ins in baseline.past_insights_by_month[ym]:
            if ins.pattern_type in seen_types_this_month:
                continue  # only count once per month even if duplicated
            seen_types_this_month.add(ins.pattern_type)
            type_counts[ins.pattern_type] = type_counts.get(ins.pattern_type, 0) + 1
            type_categories[ins.pattern_type] = ins.category
    # Need ≥ _REINFORCEMENT_PAST_MIN past-month occurrences. Current month is
    # then required to also show the pattern via _pattern_holds_in_current_month
    # below, bringing the total to ≥3 ("same outcome across 3 months" per spec).
    qualifying = [t for t, c in type_counts.items() if c >= _REINFORCEMENT_PAST_MIN]
    if not qualifying:
        return None
    qualifying.sort(key=lambda t: (-type_counts[t], t))
    chosen_type: Optional[str] = None
    for ptype in qualifying:
        if _pattern_holds_in_current_month(ptype, snapshot, phases):
            chosen_type = ptype
            break
    if chosen_type is None:
        return None
    top_cat = type_categories[chosen_type]
    if top_cat not in ("Great Job", "Need Attention", "Opportunity"):
        return None
    strength = min(1.0, type_counts[chosen_type] / 3.0 + 0.5)
    return InsightCandidate(
        pattern_family="X",
        pattern_type="reinforcement_90d",
        tentative_category=top_cat,  # type: ignore[arg-type]
        phase_focus="3",
        raw_template_key="reinforcement_90d",
        raw_signal={
            "subject": "a pattern reinforced across the last three months",
            "scope": "late_month_pattern_reinforcement_90d",
            "direction_unambiguous": "kept showing up across the last three months",
            "time_window_human": "in the closing stretch",
            "comparison_anchor": "compared to the last three months",
            "reinforced_pattern_type": chosen_type,
            "occurrences_in_last_3_months": type_counts[chosen_type],
            "carried_category": top_cat,
        },
        signal_strength=round(strength, 3),
        cross_module_impact=0.60,
    )


def _habit_180d_p3(
    snapshot: MonthlySnapshot,
    phases: List[PhaseMetrics],
    baseline: BaselineContext,
) -> Optional[InsightCandidate]:
    """Rules §1 Phase 3 row 13: 'Pattern matches long-term trend' → Great Job.

    Phase 3 variant of ``_habit_alignment_180d``: checks current month's
    overall_avg against 180d avg. Phase-focused on Phase 3 for end-of-month
    synthesis. Exempt from novelty penalty.
    """
    if not baseline.has_180d_baseline():
        return None
    cur = snapshot.overall_day_score_avg
    avg180 = baseline.avg_180d.get("overall_day_score_avg")
    if cur is None or avg180 is None or avg180 <= 0:
        return None
    delta = pct_delta(cur, avg180)
    if delta is None or abs(delta) > _HABIT_180D_MATCH_PCT:
        return None
    strength = 0.5 + (_HABIT_180D_MATCH_PCT - abs(delta)) * 2.0
    return InsightCandidate(
        pattern_family="X",
        pattern_type="habit_180d",
        tentative_category="Great Job",
        phase_focus="3",
        raw_template_key="habit_180d",
        raw_signal={
            "subject": "your day-to-day activity against the long-term habit",
            "scope": "late_month_aligned_with_180d_habit",
            "direction_unambiguous": "stayed in line with the long-term habit",
            "time_window_human": "in the closing stretch",
            "comparison_anchor": "compared to the 180-day average",
            "current_month_avg": round(cur, 2),
            "avg_180d": round(avg180, 2),
            "delta": round(delta, 3),
        },
        signal_strength=round(min(1.0, strength), 3),
        cross_module_impact=0.60,
    )


def _final_day_month_fields(snapshot: MonthlySnapshot, anchor: str) -> Dict[str, Any]:
    """Time fields shared by the 5 final-day synthesis cards.

    Final-day patterns surface ONLY in the day-1-2 window, so the analysed month
    is always the month that just ended. Naming it explicitly (e.g. "across May
    2026") lets the renderer anchor the wrap-up to that month BY NAME or call it
    "last month". ``year`` is included so the number-grounding validator accepts
    the year digits; ``month_year`` feeds the template fallback.
    """
    my = f"{month_name[snapshot.month]} {snapshot.year}"
    return {
        "time_window_human": f"{anchor} {my}",
        "month_year": my,
        "year": snapshot.year,
    }


def _historical_comparison_final_day(
    snapshot: MonthlySnapshot,
    phases: List[PhaseMetrics],
    baseline: BaselineContext,
) -> Optional[InsightCandidate]:
    """Rules §1 Final-Day row 5: 'Month vs 90d / 180d compare outcomes' → All buckets.

    Compares the month's overall day score against the recent baseline (90d,
    falling back to 180d) and routes by direction: above baseline → Great Job,
    below → Opportunity (framed as potential per §3 guard). Needs a near-complete
    month (``_FINAL_DAY_MIN_DAYS``) and at least one baseline; skipped otherwise.
    """
    if snapshot.days_with_data < _FINAL_DAY_MIN_DAYS:
        return None
    base = baseline.avg_90d.get("overall_day_score_avg")
    base_label = "90-day"
    if base is None:
        base = baseline.avg_180d.get("overall_day_score_avg")
        base_label = "180-day"
    cur = snapshot.overall_day_score_avg
    if cur is None or base is None or base <= 0:
        return None
    delta = pct_delta(cur, base)
    if delta is None or abs(delta) < _DIVERGENCE_VS_HISTORY_PCT:
        return None
    direction = "above" if delta > 0 else "below"
    # Route across ALL buckets (spec FD-row-5 "All"): above baseline → Great Job;
    # a mild drop → Opportunity (frame as potential per §3 guard); a steep drop
    # (≥ _FINAL_DAY_HISTCOMP_NA_DROP below) → Need Attention.
    if delta > 0:
        category = "Great Job"
    elif delta <= -_FINAL_DAY_HISTCOMP_NA_DROP:
        category = "Need Attention"
    else:
        category = "Opportunity"
    strength = min(1.0, abs(delta) / 0.40 + 0.3)
    raw_signal: Dict[str, Any] = {
        "subject": "your day-to-day activity against the recent baseline",
        "scope": "final_day_month_vs_baseline",
        "direction_unambiguous": (
            "finished above the recent baseline"
            if delta > 0
            else "finished below the recent baseline"
        ),
        **_final_day_month_fields(snapshot, "across"),
        "comparison_anchor": f"compared to the {base_label} average",
        # Numeric fields kept for scoring/audit; stripped from the LLM view
        # via _INTERNAL_RAW_SIGNAL_KEYS so no raw score/delta leaks.
        "current_month_avg": round(cur, 2),
        "delta": round(delta, 3),
        "direction": direction,
    }
    # Below-baseline cards (Opportunity / Need Attention) carry a soft suggestion
    # the renderer can voice; above-baseline (Great Job) needs none.
    if delta < 0:
        raw_signal["improvement_direction"] = (
            "bringing your daily rhythm back toward its recent baseline"
        )
    return InsightCandidate(
        pattern_family="X",
        pattern_type="historical_comparison_final_day",
        tentative_category=category,  # type: ignore[arg-type]
        phase_focus="cross",
        raw_template_key="historical_comparison_final_day",
        raw_signal=raw_signal,
        signal_strength=round(strength, 3),
        cross_module_impact=0.50,
    )


def _dominant_pattern_final_day(
    snapshot: MonthlySnapshot, phases: List[PhaseMetrics]
) -> Optional[InsightCandidate]:
    """Rules §1 Final-Day row 1: 'Strongest signal across all phases' → All buckets.

    Spec gives one line ("strongest signal across all phases"); the engine
    architecture does not expose other detectors' outputs to a single detector,
    so this implementation surveys the three signal families that map directly
    to Rules §1 patterns (imbalance, correlation trade-off, phase trend) and
    emits the one with the highest salience (magnitude relative to its OWN
    trigger floor). Other detectors still fire independently for their
    respective categories — this detector exists only to satisfy the FD-row-1
    "synthesis" wording.

    Category derivation per Rules §3 "Category Assignment":
      - Negative correlation → Need Attention (trade-off)
      - Positive correlation → Great Job (positive link)
      - Imbalance or phase trend → Opportunity (frame as potential)
    """
    if snapshot.days_with_data < _FINAL_DAY_MIN_DAYS:
        return None

    # Each entry: (kind, salience, magnitude, payload). ``salience`` = magnitude
    # relative to that signal's OWN trigger floor, so the three kinds (different
    # units: |corr|, imbalance excess, trend pct-delta) compare fairly when
    # picking the strongest.
    candidates: List[Tuple[str, float, float, Dict[str, Any]]] = []

    # (a) Module imbalance
    excess = snapshot.module_max_excess_pct
    area = snapshot.most_concentrated_area
    if (
        excess is not None
        and area is not None
        and excess >= _FINAL_DAY_DOMINANT_MIN_EXCESS
    ):
        candidates.append(
            (
                "imbalance",
                float(excess) / _FINAL_DAY_DOMINANT_MIN_EXCESS,
                float(excess),
                {
                    "signal_kind": "imbalance",
                    "area": str(area),
                    "excess": round(excess, 3),
                },
            )
        )

    # (b) Cross-module correlation — reuses _TRADEOFF_PAIR_LABEL to avoid
    # duplicating the readable labels.
    corr_lookup: List[Tuple[str, Optional[float]]] = [
        ("work_sleep", snapshot.corr_work_sleep),
        ("work_health", snapshot.corr_work_health),
        ("work_spend", snapshot.corr_work_spend),
        ("sleep_mood", snapshot.corr_sleep_mood),
        ("workout_mood", snapshot.corr_workout_mood),
        ("meeting_mood", snapshot.corr_meeting_mood),
    ]
    corr_pool = [(name, c) for name, c in corr_lookup if c is not None]
    if corr_pool:
        name, corr = max(corr_pool, key=lambda nc: abs(nc[1]))
        if abs(corr) >= _FINAL_DAY_TRADEOFF_CORR:
            label = _TRADEOFF_PAIR_LABEL.get(name, name.replace("_", " and "))
            candidates.append(
                (
                    "correlation",
                    abs(corr) / _FINAL_DAY_TRADEOFF_CORR,
                    abs(corr),
                    {
                        "signal_kind": "correlation",
                        "pair": label,
                        "direction": "negative" if corr < 0 else "positive",
                        "strength_label": _strength_label(corr),
                    },
                )
            )

    # (c) Phase-to-phase trend (largest absolute delta among P1→P2 and P2→P3)
    p1_s = phases[0].overall_day_score_avg
    p2_s = phases[1].overall_day_score_avg
    p3_s = phases[2].overall_day_score_avg
    trend_pool: List[Tuple[str, float]] = []
    d21 = pct_delta(p2_s, p1_s)
    if d21 is not None:
        trend_pool.append(("early to mid-month", d21))
    d32 = pct_delta(p3_s, p2_s)
    if d32 is not None:
        trend_pool.append(("mid-month to end of month", d32))
    if trend_pool:
        leg, delta = max(trend_pool, key=lambda ld: abs(ld[1]))
        if abs(delta) >= _LEVEL_VS_HISTORY_PCT:
            candidates.append(
                (
                    "trend",
                    abs(delta) / _LEVEL_VS_HISTORY_PCT,
                    abs(delta),
                    {
                        "signal_kind": "trend",
                        "leg": leg,
                        "direction": "increasing" if delta > 0 else "decreasing",
                    },
                )
            )

    if not candidates:
        return None
    # Pick by salience (relative to each signal's own floor), not raw magnitude.
    kind, _salience, magnitude, payload = max(candidates, key=lambda c: c[1])

    # Category derived purely from signal nature (Rules §3 + Rules §1 Final-Day
    # row 1 "All buckets") — no absolute-score gate.
    category: str
    direction = payload.get("direction")
    if kind == "correlation" and direction == "negative":
        category = "Need Attention"
    elif kind == "correlation" and direction == "positive":
        category = "Great Job"
    else:
        # imbalance, trend (either direction) → Opportunity per §3 guard
        # ("Frame as potential, not deficiency").
        category = "Opportunity"

    # Build a unified ``area`` filler so the template stays single-line.
    if kind == "imbalance":
        subject = f"a concentration in {payload['area']}"
    elif kind == "correlation":
        subject = f"the relationship between {payload['pair']}"
    else:
        subject = f"the shift from {payload['leg']}"
    strength = min(1.0, magnitude / 0.60 + 0.4)
    # Surface a qualitative strength_label (STRING) instead of any raw magnitude
    # number, so the LLM has no decimal to leak and the numeric validator no
    # number to misread.
    payload_for_llm = dict(payload)
    payload_for_llm["strength_label"] = _strength_label(magnitude)
    return InsightCandidate(
        pattern_family="X",
        pattern_type="dominant_pattern_final_day",
        tentative_category=category,  # type: ignore[arg-type]
        phase_focus="cross",
        raw_template_key="dominant_pattern_final_day",
        raw_signal={
            "subject": f"the strongest signal of the month — {subject}",
            "scope": "final_day_dominant_signal",
            # Spec FD-row-1 output names the month ("In April 2026, …").
            **_final_day_month_fields(snapshot, "in"),
            **payload_for_llm,
            "area": subject,
        },
        signal_strength=round(strength, 3),
        cross_module_impact=0.75,
    )


# Correlation-pair internal name → user-readable phrase (for trade-off summary).
_TRADEOFF_PAIR_LABEL: Dict[str, str] = {
    "work_sleep": "workload and sleep",
    "work_health": "workload and recovery-related activity",
    "work_spend": "workload and spending",
    "sleep_mood": "sleep and mood",
    "workout_mood": "workouts and mood",
    "meeting_mood": "meeting load and mood",
}


def _strength_label(magnitude: float) -> str:
    """Map |magnitude| ∈ [0, 1] to a qualitative English descriptor.

    Used to surface "strong" / "moderate" / "mild" to the LLM as a STRING so
    the renderer can describe the signal qualitatively without ever seeing a
    raw correlation coefficient or fraction. The English label is universal
    (the LLM translates it to the user's language) and the validator never
    has a number to misinterpret.
    """
    m = abs(magnitude)
    if m >= 0.70:
        return "strong"
    if m >= 0.50:
        return "moderate"
    return "mild"


# ---------------------------------------------------------------------------
# Qualitative label helpers — replace raw numbers in ``raw_signal`` so the
# LLM renders observations conversationally instead of citing "0.45 of days"
# or "5.8 hours". Each helper returns a fixed English string the renderer can
# embed directly; the validator never sees a number it might mis-render.
# ---------------------------------------------------------------------------


def _label_position_0_100(score: float) -> str:
    """Position of a module composite score on the 0..100 scale."""
    if score < 30:
        return "low"
    if score < 60:
        return "moderate"
    if score < 80:
        return "good"
    return "strong"


def _label_frequency(fraction: float) -> str:
    """Fraction of days a behaviour occurs (0..1)."""
    if fraction < 0.20:
        return "rare"
    if fraction < 0.40:
        return "occasional"
    if fraction < 0.60:
        return "frequent"
    return "very_frequent"


def _label_sleep_hours(hours: float) -> str:
    """Average sleep hours — qualitative."""
    if hours < 6.0:
        return "well_below_target"
    if hours < 7.0:
        return "below_target"
    if hours < 9.0:
        return "in_range"
    return "above_target"


def _label_balance_drop(points: float) -> str:
    """Mean-based balance drop on 0..100 scale, in absolute points."""
    a = abs(points)
    if a < 10:
        return "slight"
    if a < 20:
        return "moderate"
    if a < 30:
        return "significant"
    return "sharp"


def _label_imbalance_std(std: float) -> str:
    """Stddev of 3 module scores (0..100 scale) — measures spread."""
    if std < 10:
        return "even"
    if std < 20:
        return "uneven"
    return "very_uneven"


def _label_mood_range(range_avg: float) -> str:
    """Mood daily range mean — m_mood_score_avg is 1..5 so max range is 4."""
    if range_avg < 1.0:
        return "stable"
    if range_avg < 2.0:
        return "variable"
    return "volatile"


def _label_event_count(events_per_day: float) -> str:
    """Calendar density — events per day."""
    if events_per_day < 3:
        return "light"
    if events_per_day < 5:
        return "moderate"
    if events_per_day < 8:
        return "busy"
    return "very_busy"


def _label_meeting_load(meeting_min_avg: float) -> str:
    """Average daily meeting minutes."""
    if meeting_min_avg < 60:
        return "light"
    if meeting_min_avg < 180:
        return "moderate"
    if meeting_min_avg < 360:
        return "heavy"
    return "very_heavy"


def _label_alignment_pct(fraction: float) -> str:
    """Goal-vs-reality alignment fraction (0..1+). Replaces raw % in raw_signal."""
    if fraction < 0.30:
        return "off_track"
    if fraction < 0.60:
        return "lagging"
    if fraction < 0.90:
        return "on_track"
    if fraction <= 1.10:
        return "at_goal"
    return "exceeding"


# -------- concrete anchor helpers (Cải tiến 3 / Hướng B) -----------------
# These helpers add distinctive concrete fields to ``raw_signal`` so the LLM
# can write distinguishable text for patterns that previously paraphrased to
# the same vocabulary (e.g. ``sustained_balance`` and ``stable_rhythm_p2``
# both becoming "ổn định").


def _label_balance_strength(avg_balance: float) -> str:
    """Categorise a 0..100 balance score above the 75 floor into a strength
    band. Used by sustained_balance to give the LLM a distinct anchor word."""
    if avg_balance >= 90:
        return "very_strong"
    if avg_balance >= 82:
        return "strong"
    return "solid"


def _pick_top_module(
    module_scores: Optional[Dict[str, float]],
) -> Optional[str]:
    """Return name of the highest-scoring module from ``module_scores`` dict
    (e.g. {"health": 80, "productivity": 75, "finance": 70} → "health").
    Returns None when the dict is empty / missing."""
    if not module_scores:
        return None
    return max(module_scores, key=module_scores.get)


def _pick_supporting_modules(
    module_scores: Optional[Dict[str, float]],
    top: Optional[str],
) -> List[str]:
    """Return the remaining module names (excluding ``top``) ordered by score
    desc. Used so the LLM can name the dominant module AND its companions."""
    if not module_scores or top is None:
        return []
    others = [(m, s) for m, s in module_scores.items() if m != top]
    others.sort(key=lambda ms: ms[1], reverse=True)
    return [m for m, _ in others]


def _label_timing_std_minutes(std_min: float) -> str:
    """Bedtime/wake-time stddev in minutes — qualitative band."""
    if std_min < 30:
        return "very_steady"
    if std_min < 60:
        return "steady"
    if std_min < 90:
        return "noticeable"
    return "wide"


def _pick_variability_source(
    phases: List[PhaseMetrics],
    snapshot: MonthlySnapshot,
) -> Optional[str]:
    """Identify whether routine variability comes from sleep hours OR overall
    daily-score variance. Used by irregular_timing / consistent_routine to
    give the LLM a concrete metric name rather than a generic "schedule"."""
    sleep_stds = [
        p.h_sleep_hours_std for p in phases if p.h_sleep_hours_std is not None
    ]
    sleep_worst = max(sleep_stds) if sleep_stds else 0.0
    overall_std = snapshot.overall_day_score_std or 0.0
    if not sleep_stds and overall_std == 0.0:
        return None
    if sleep_stds and overall_std == 0.0:
        return "sleep_hours"
    if not sleep_stds and overall_std > 0:
        return "daily_activity"
    # Both present — pick which is more noteworthy.
    # Normalise: sleep_std >= 1.0 hour is "noisy"; overall_std >= 12 is "noisy".
    sleep_norm = sleep_worst / 1.0
    overall_norm = overall_std / 12.0
    if abs(sleep_norm - overall_norm) < 0.2:
        return "both"
    return "sleep_hours" if sleep_norm > overall_norm else "daily_activity"


def _tradeoff_summary_final_day(
    snapshot: MonthlySnapshot, phases: List[PhaseMetrics]
) -> Optional[InsightCandidate]:
    """Rules §1 Final-Day row 2: 'Most significant imbalance — highest correlation across modules' → Need Attention.

    Picks the strongest NEGATIVE correlation across all module pairs (the actual
    trade-off) and frames it as the "strongest pattern this month". Positive
    correlations are beneficial, not trade-offs, so they are ignored here.
    """
    if snapshot.days_with_data < _FINAL_DAY_MIN_DAYS:
        return None
    corr_pairs: List[Tuple[str, Optional[float]]] = [
        ("work_sleep", snapshot.corr_work_sleep),
        ("work_health", snapshot.corr_work_health),
        ("work_spend", snapshot.corr_work_spend),
        ("sleep_mood", snapshot.corr_sleep_mood),
        ("workout_mood", snapshot.corr_workout_mood),
        ("meeting_mood", snapshot.corr_meeting_mood),
    ]
    # A trade-off is a NEGATIVE relationship (one area rises as another falls);
    # only the strongest negative link qualifies. Positive correlations are
    # beneficial and surface elsewhere (dominant_pattern → Great Job, Family C/F).
    negatives = [(name, c) for name, c in corr_pairs if c is not None and c < 0]
    if not negatives:
        return None
    name, corr = min(negatives, key=lambda nc: nc[1])  # most negative
    if corr > -_FINAL_DAY_TRADEOFF_CORR:  # |corr| below the trade-off floor
        return None
    pair_label = _TRADEOFF_PAIR_LABEL.get(name, name.replace("_", " and "))
    strength = min(1.0, abs(corr))
    return InsightCandidate(
        pattern_family="X",
        pattern_type="tradeoff_summary_final_day",
        tentative_category="Need Attention",
        phase_focus="cross",
        raw_template_key="tradeoff_summary_final_day",
        # Qualitative strength_label keeps the LLM grounded without exposing
        # a raw correlation coefficient (Rules §0 row 7 + §4).
        raw_signal={
            "subject": f"the trade-off between {pair_label}",
            "scope": "final_day_strongest_tradeoff",
            "direction_unambiguous": "moved in opposite directions",
            **_final_day_month_fields(snapshot, "from start to finish in"),
            "improvement_direction": f"loosening the link between {pair_label}",
            "pair": pair_label,
            "direction": "negative",
            "strength_label": _strength_label(corr),
        },
        signal_strength=round(strength, 3),
        cross_module_impact=1.0,
    )


def _stability_summary_final_day(
    snapshot: MonthlySnapshot, phases: List[PhaseMetrics]
) -> Optional[InsightCandidate]:
    """Rules §1 Final-Day row 3: 'Consistent behaviour — low variance across phases' → Great Job.

    Distinct from ``_stable_rhythm`` (P3 row 2) by:
      - Stricter min-days gate (`_FINAL_DAY_MIN_DAYS`)
      - Final-day retrospective wording ("You maintained a steady pattern…")
    """
    if snapshot.days_with_data < _FINAL_DAY_MIN_DAYS:
        return None
    scores = [
        p.overall_day_score_avg for p in phases if p.overall_day_score_avg is not None
    ]
    if len(scores) != 3:
        return None
    lo, hi = min(scores), max(scores)
    if lo == 0:
        return None
    spread = (hi - lo) / lo
    if spread > _FINAL_DAY_STABILITY_MAX_SPREAD:
        return None
    strength = min(
        1.0,
        (_FINAL_DAY_STABILITY_MAX_SPREAD - spread) / _FINAL_DAY_STABILITY_MAX_SPREAD
        + 0.5,
    )
    return InsightCandidate(
        pattern_family="X",
        pattern_type="stability_summary_final_day",
        tentative_category="Great Job",
        phase_focus="cross",
        raw_template_key="stability_summary_final_day",
        raw_signal={
            "subject": "your day-to-day activity from start to finish",
            "scope": "final_day_low_variance_across_phases",
            "direction_unambiguous": "stayed steady",
            **_final_day_month_fields(snapshot, "all the way through"),
            "phase_score_spread_pct": round(spread, 3),
        },
        signal_strength=round(strength, 3),
        cross_module_impact=0.60,
    )


def _improvement_area_final_day(
    snapshot: MonthlySnapshot,
    phases: List[PhaseMetrics],
    baseline: BaselineContext,
) -> Optional[InsightCandidate]:
    """Rules §1 Final-Day row 4: 'Largest deviation — biggest gap vs baseline' → Opportunity.

    Surfaces the module that fell FURTHEST below its OWN recent baseline (90d,
    fallback 180d) as the area with room to improve. Fires only when that gap
    clears ``_FINAL_DAY_IMPROVEMENT_MIN_GAP`` so it stays quiet when every module
    sits near its baseline.
    """
    if snapshot.days_with_data < _FINAL_DAY_MIN_DAYS:
        return None
    # Per-module: month value vs that module's own 90d (fallback 180d) baseline.
    module_specs: List[Tuple[str, Optional[float], str]] = [
        ("health", snapshot.h_health_score_avg, "h_health_score_avg"),
        (
            "productivity",
            snapshot.p_task_completion_rate_avg,
            "p_task_completion_rate_avg",
        ),
        ("finance", snapshot.financial_health_score_avg, "financial_health_score_avg"),
    ]
    gaps: List[Tuple[str, float]] = []
    for name, cur, key in module_specs:
        base = baseline.avg_90d.get(key) or baseline.avg_180d.get(key)
        if cur is None or base is None or base <= 0:
            continue
        gaps.append((name, (base - cur) / base))
    if not gaps:
        return None
    weakest_module, gap = max(gaps, key=lambda ng: ng[1])
    if gap < _FINAL_DAY_IMPROVEMENT_MIN_GAP:
        return None
    strength = min(1.0, gap / 0.40 + 0.3)
    return InsightCandidate(
        pattern_family="X",
        pattern_type="improvement_area_final_day",
        tentative_category="Opportunity",
        phase_focus="cross",
        raw_template_key="improvement_area_final_day",
        raw_signal={
            "subject": f"your biggest gap vs baseline this month ({weakest_module})",
            "scope": "final_day_largest_gap_vs_baseline",
            "direction_unambiguous": "trailed its recent baseline",
            **_final_day_month_fields(snapshot, "over the course of"),
            "comparison_anchor": "compared to the 90-day or 180-day average",
            "improvement_direction": f"lifting {weakest_module} closer to its usual baseline",
            "area": str(weakest_module),
            "gap_vs_baseline_pct": round(gap, 3),
        },
        signal_strength=round(strength, 3),
        cross_module_impact=0.75,
    )


def _detect_historical(
    snapshot: MonthlySnapshot,
    phases: List[PhaseMetrics],
    baseline: BaselineContext,
) -> List[InsightCandidate]:
    """Run every historical detector — all rely on ``BaselineContext`` fields.

    Detectors covering Rules §1 Phase 1 rows 1-3, 8-11; Phase 2 rows 3, 12-13;
    Phase 3 rows 10-13; Final-Day rows 4-5.
    """
    out: List[InsightCandidate] = []
    for fn in (
        _higher_than_usual_start,
        _lower_than_usual_start,
        _change_vs_last_month_start,
        _flat_start_vs_90d,
        _positive_carry_over,
        _negative_carry_over,
        _early_tradeoff_vs_baseline,
        _habit_alignment_180d,
        _divergence_vs_history_p2,
        _pattern_repeat_90d,
        _behaviour_shift_180d,
        _recovery_rebound,
        _improvement_vs_last_month,
        _deterioration_vs_last_month,
        _reinforcement_90d,
        _habit_180d_p3,
        _historical_comparison_final_day,
        _improvement_area_final_day,
    ):
        try:
            c = fn(snapshot, phases, baseline)
        except Exception as e:
            logger.warning(f"⚠️ Historical detector {fn.__name__} failed: {e}")
            continue
        if c:
            out.append(c)
    return out


# =============================================================================
# Aggregator — every detector family
# =============================================================================


def detect_all(
    snapshot: MonthlySnapshot,
    phases: List[PhaseMetrics],
    baseline: Optional[BaselineContext] = None,
) -> List[InsightCandidate]:
    out: List[InsightCandidate] = []
    out.extend(_detect_family_a(snapshot, phases))
    out.extend(_detect_family_b(snapshot, phases))
    out.extend(_detect_family_c(snapshot, phases))
    out.extend(_detect_family_d(snapshot, phases))
    out.extend(_detect_family_e(snapshot, phases))
    out.extend(_detect_family_f(snapshot, phases))
    out.extend(_detect_family_x(snapshot, phases))
    # Rules §1 historical detectors — only fire when baseline data is present.
    if baseline is not None:
        out.extend(_detect_historical(snapshot, phases, baseline))
    return out


# =============================================================================
# Intelligence layers — see ``docs/INSIGHT_DETAILED_REPORT.md``
# -----------------------------------------------------------------------------
# Sits between detector output and rendering. Adds context-aware
# interpretation on top of the pattern engine without rewriting detectors:
#
#   [2]  Quality Gates           — suppress patterns from sparse/corrupt data
#   [3]  Gradient floor          — global min-strength after detection
#   [5]  Context Interpretation  — re-route category by life_mode + goal difficulty
#   [6]  Confidence Scoring      — completeness × consistency × signal × repeatability
#   [9]  Memory Annotation       — sustained / escalation / recovery_cycle markers
#   [10] Quality Validator       — reject generic / repetitive / over-confident
#   [11] Tone Hints              — confidence_tier + memory feed render tone
#
# Layers are pure functions; they communicate by extending
# ``InsightCandidate.raw_signal`` because the candidate model is forbid-extra.
# Feature-flagged where they change observable category routing so the engine
# can ship without behaviour drift — flip flags after empirical validation.
# =============================================================================


# ----- Layer [2] thresholds -------------------------------------------------

# Min strength a candidate must clear to survive volume-floor filtering.
# Ramps up as days_with_data drops — sparser month, stronger evidence required.
_QG_FLOOR_HEAVY: float = 0.20  # ≥14 days
_QG_FLOOR_LIGHT: float = 0.40  # 7-13 days
_QG_FLOOR_SPARSE: float = 0.60  # <7 days

# Coverage below which a module is too sparse to claim against.
_QG_SPARSE_MODULE_COVERAGE: float = 0.30

# Identical mood across many days is almost always a sync bug, not a user
# with literally constant mood (Feedback §9). Suppress mood patterns when set.
_QG_MOOD_STUCK_MIN_DAYS: int = 20

# Modules tracked end-to-end in the engine — mirrors rollup fields on MonthlySnapshot.
_TRACKED_MODULES: Tuple[str, ...] = ("health", "productivity", "finance", "mood")

# Per-module snapshot field that, when ``None``, indicates the module has no
# usable data this month. Read by ``_module_coverage``. Field names match
# ``MonthlySnapshot`` — productivity uses task completion rate as proxy because
# the snapshot model does not carry a unified ``productivity_score_avg``.
_MODULE_AVG_FIELD: Dict[str, str] = {
    "health": "h_health_score_avg",
    "productivity": "p_task_completion_rate_avg",
    "finance": "financial_health_score_avg",
    "mood": "m_mood_score_avg",
}

# Which modules each pattern primarily reads. A pattern is dropped when ALL
# of its modules are sparse, OR when mood-stuck trips and any of its modules
# is ``mood``. Patterns absent from this map are never suppressed by Quality
# Gates — they likely depend on cross-phase / multi-module signals not tied
# to a single rollup field.
_PATTERN_MODULES: Dict[str, Tuple[str, ...]] = {
    # A. Calendar
    "meeting_overload": ("productivity", "health"),
    "focused_work_blocks": ("productivity",),
    "context_switching": ("productivity",),
    "social_energy_days": ("productivity", "mood"),
    # B. Time-of-day
    "late_work_impact": ("productivity", "health"),
    "consistent_routine": ("health",),
    "irregular_timing": ("health",),
    # C. Sequence
    "sleep_work_chain": ("health", "productivity"),
    "workout_mood_chain": ("health", "mood"),
    "busy_day_spending": ("productivity", "finance"),
    # D. Goals — alignment cuts across all goal-tracked modules
    "consistent_goal_alignment": ("health", "productivity", "finance"),
    "partial_alignment": ("health", "productivity", "finance"),
    "misalignment": ("health", "productivity", "finance"),
    # F. Finance
    "stress_linked_spending": ("productivity", "finance"),
    "event_driven_spending": ("productivity", "finance"),
    "controlled_spending": ("productivity", "finance"),
    # X. Mood / cross-module
    "mood_volatility": ("mood",),
    "stress_pattern": ("productivity", "health"),
    "finance_imbalance": ("finance",),
    "work_health_tradeoff_corr": ("productivity", "health"),
    "finance_work_link": ("productivity", "finance"),
}

# ----- Layer [3] gradient floor ---------------------------------------------

# Per Feedback §5: catches anything that slips through with very weak strength
# regardless of detector-level thresholds. Per-detector formula refactor is
# tracked in INSIGHT_PIPELINE.md as future work.
_GRADIENT_NOISE_FLOOR: float = 0.20

# ----- Layer [6] confidence weights -----------------------------------------

# Feedback Priority 3 formula. Sum of weights = 1.0.
_CONF_W_COMPLETENESS: float = 0.30
_CONF_W_CONSISTENCY: float = 0.20
_CONF_W_SIGNAL: float = 0.30
_CONF_W_REPEATABILITY: float = 0.20

_CONF_TIER_HIGH: float = 0.70
_CONF_TIER_MODERATE: float = 0.50

# ----- Layer [5] context thresholds -----------------------------------------

_CTX_HIGH_GROWTH_TREND: float = 0.15
_CTX_MEDIUM_GROWTH_TREND: float = 0.05
_CTX_HIGH_ALIGNMENT_FLOOR: float = 0.80

_CTX_TRANSITION_DELTA: float = 0.30
_CTX_INTENSITY_PROD_DELTA: float = 0.15
_CTX_RECOVERY_HEALTH_DELTA: float = -0.15
_CTX_RECOVERY_PROD_DELTA: float = -0.10
_CTX_GROWTH_PROD_DELTA: float = 0.10
_CTX_GROWTH_HEALTH_BAND: float = 0.10
_CTX_MOOD_STABLE_STD: float = 0.6

# ----- Layer [10] quality validator -----------------------------------------

# Literal values from Feedback §6 "Example Rules".
_QV_SIGNAL_FLOOR: float = 0.35
_QV_TEXT_SIMILARITY_CEIL: float = 0.80
_QV_LOW_CONF_DOWNGRADE_CEIL: float = 0.50


# =============================================================================
# Layer [2] Quality Gates
# =============================================================================


@dataclass(frozen=True)
class QualityFlags:
    """Output of Layer [2] — diagnostics consumed by later layers.

    ``sparse_modules`` — modules whose coverage this month is below
        ``_QG_SPARSE_MODULE_COVERAGE``. Patterns reading only sparse
        modules are dropped by ``apply_quality_gates``.
    ``mood_stuck`` — set when mood std=0 across ≥20 days, the canonical
        signature of a sync-pipeline bug. Any pattern touching mood is
        suppressed when this trips.
    ``min_strength_floor`` — raised when the month has few days of data so
        only stronger signals reach the user.
    """

    sparse_modules: frozenset
    mood_stuck: bool
    min_strength_floor: float


def _module_coverage(snapshot: MonthlySnapshot, module: str) -> float:
    """Conservative module-coverage estimate.

    We don't have per-module day counts yet, so coverage is approximated as:
    a module is "covered" iff its rollup average is populated, capped by
    total ``days_with_data``. Errs on the side of declaring modules sparse
    when in doubt — the safer default for confidence claims.
    """
    days = snapshot.days_with_data or 0
    if days <= 0:
        return 0.0
    field_name = _MODULE_AVG_FIELD.get(module)
    if field_name is None:
        return 1.0
    if getattr(snapshot, field_name, None) is None:
        return 0.0
    return min(1.0, days / 30.0)


def _phase_mood_stds(phases: List[PhaseMetrics]) -> List[float]:
    """Return populated per-phase mood std-devs (the snapshot model carries no monthly std)."""
    return [p.m_mood_score_std for p in phases if p.m_mood_score_std is not None]


def compute_quality_flags(
    snapshot: MonthlySnapshot, phases: List[PhaseMetrics]
) -> QualityFlags:
    """Build QualityFlags for the analysed month."""
    days = snapshot.days_with_data or 0

    if days < 7:
        floor = _QG_FLOOR_SPARSE
    elif days < 14:
        floor = _QG_FLOOR_LIGHT
    else:
        floor = _QG_FLOOR_HEAVY

    sparse: List[str] = [
        m
        for m in _TRACKED_MODULES
        if _module_coverage(snapshot, m) < _QG_SPARSE_MODULE_COVERAGE
    ]

    # Mood-stuck check needs std data. MonthlySnapshot doesn't carry it, so we
    # aggregate from PhaseMetrics — stuck = every populated phase shows std=0.
    phase_stds = _phase_mood_stds(phases)
    mood_stuck = (
        bool(phase_stds)
        and all(s == 0 for s in phase_stds)
        and days >= _QG_MOOD_STUCK_MIN_DAYS
    )

    return QualityFlags(
        sparse_modules=frozenset(sparse),
        mood_stuck=mood_stuck,
        min_strength_floor=floor,
    )


def _pattern_blocked_by_data(candidate: InsightCandidate, flags: QualityFlags) -> bool:
    """Decide whether a candidate is reading too-sparse/corrupt data to keep."""
    modules = _PATTERN_MODULES.get(candidate.pattern_type)
    if not modules:
        return False
    if flags.mood_stuck and "mood" in modules:
        return True
    # Drop only when EVERY module this pattern reads is sparse. A multi-module
    # pattern can survive on one good module.
    return all(m in flags.sparse_modules for m in modules)


def apply_quality_gates(
    candidates: List[InsightCandidate], flags: QualityFlags
) -> List[InsightCandidate]:
    """Drop candidates lacking trustworthy data backing."""
    return [
        c
        for c in candidates
        if not _pattern_blocked_by_data(c, flags)
        and c.signal_strength >= flags.min_strength_floor
    ]


# =============================================================================
# Layer [3] Gradient floor
# =============================================================================


def apply_gradient_floor(
    candidates: List[InsightCandidate],
) -> List[InsightCandidate]:
    """Drop candidates below the global noise floor.

    Per Feedback §5, individual detectors should eventually replace hard
    cutoffs with continuous strength formulas. Until that per-detector
    refactor lands, this single chokepoint catches anything too weak to
    be worth surfacing to the user.
    """
    return [c for c in candidates if c.signal_strength >= _GRADIENT_NOISE_FLOOR]


# =============================================================================
# Layer [5] Context Interpretation
# =============================================================================


@dataclass(frozen=True)
class Context:
    """Behavioural context derived for the analysed month.

    Used to re-route ``tentative_category`` away from the cliff-edged mapping
    detectors produce — implements Feedback §1 (stagnation reframe) and §4
    (intentional intensity reframe). All four fields are also stamped on
    raw_signal for audit / downstream tone hinting.
    """

    life_mode: str  # STABLE / GROWTH / INTENSIVE / RECOVERY / TRANSITION
    goal_difficulty: str  # low / medium / high
    growth_trend: float  # alignment delta vs prev month, [-1, +1]
    intentional_intensity: bool


def _safe_pct_delta(
    current: Optional[float], baseline: Optional[float]
) -> Optional[float]:
    """``(current - baseline) / baseline`` with None / zero-division guards."""
    if current is None or baseline is None or baseline == 0:
        return None
    return (current - baseline) / baseline


def _mean_alignment(snapshot: Optional[MonthlySnapshot]) -> Optional[float]:
    """Average of available goal-alignment percentages, normalised to [0, 1]."""
    if snapshot is None:
        return None
    values = [
        snapshot.goal_alignment_health_pct,
        snapshot.goal_alignment_productivity_pct,
        snapshot.goal_alignment_finance_pct,
    ]
    present = [v for v in values if v is not None]
    if not present:
        return None
    return sum(present) / len(present) / 100.0


def _classify_life_mode(
    snapshot: MonthlySnapshot,
    phases: List[PhaseMetrics],
    baseline: BaselineContext,
) -> str:
    """Pick one of STABLE / GROWTH / INTENSIVE / RECOVERY / TRANSITION.

    Mirrors the heuristic in ``INSIGHT_PIPELINE.md`` §[5]:
        TRANSITION  : month differs from 180d baseline by ≥30% (either direction)
        RECOVERY    : sustained multi-month decline in health AND productivity
        INTENSIVE   : productivity spike + mood stable + visible P3 rebound
        GROWTH      : productivity up ~10% while health stays within ±10%
        STABLE      : default
    """
    avg_180 = baseline.avg_180d if baseline.has_180d_baseline() else {}
    avg_90 = baseline.avg_90d if baseline.has_90d_baseline() else {}

    overall_delta_180 = _safe_pct_delta(
        snapshot.overall_day_score_avg, avg_180.get("overall_day_score_avg")
    )
    if (
        overall_delta_180 is not None
        and abs(overall_delta_180) >= _CTX_TRANSITION_DELTA
    ):
        return "TRANSITION"

    health_delta_180 = _safe_pct_delta(
        snapshot.h_health_score_avg, avg_180.get("h_health_score_avg")
    )
    # Productivity baseline travels under the ``p_task_completion_rate_avg``
    # key in BaselineContext (see ``_BASELINE_METRIC_KEYS``), and the snapshot
    # tracks the same field directly — keep both sides aligned.
    prod_delta_90 = _safe_pct_delta(
        snapshot.p_task_completion_rate_avg, avg_90.get("p_task_completion_rate_avg")
    )

    # Recovery: this month low + previous month also low → sustained dip,
    # not a one-month blip.
    prev = baseline.prev_month_snapshot
    if prev is not None and health_delta_180 is not None and prod_delta_90 is not None:
        prev_health_delta = _safe_pct_delta(
            prev.h_health_score_avg, avg_180.get("h_health_score_avg")
        )
        sustained_down = (
            health_delta_180 <= _CTX_RECOVERY_HEALTH_DELTA
            and prev_health_delta is not None
            and prev_health_delta <= -0.10
        )
        if sustained_down and prod_delta_90 <= _CTX_RECOVERY_PROD_DELTA:
            return "RECOVERY"

    # Intensive: productivity spike + mood stable + visible P3 rebound.
    # Snapshot lacks a monthly mood std — aggregate from phases instead.
    phase_mood_stds = _phase_mood_stds(phases)
    mood_stable = bool(phase_mood_stds) and (
        sum(phase_mood_stds) / len(phase_mood_stds) <= _CTX_MOOD_STABLE_STD
    )
    has_rebound = (
        len(phases) == 3
        and phases[0].overall_day_score_avg is not None
        and phases[1].overall_day_score_avg is not None
        and phases[2].overall_day_score_avg is not None
        and phases[1].overall_day_score_avg < phases[0].overall_day_score_avg
        and phases[2].overall_day_score_avg > phases[1].overall_day_score_avg
    )
    if (
        prod_delta_90 is not None
        and prod_delta_90 >= _CTX_INTENSITY_PROD_DELTA
        and mood_stable
        and has_rebound
    ):
        return "INTENSIVE"

    # Growth: productivity up while health stays in band
    if (
        prod_delta_90 is not None
        and prod_delta_90 >= _CTX_GROWTH_PROD_DELTA
        and (
            health_delta_180 is None or abs(health_delta_180) <= _CTX_GROWTH_HEALTH_BAND
        )
    ):
        return "GROWTH"

    return "STABLE"


def _compute_growth_trend(
    snapshot: MonthlySnapshot, baseline: BaselineContext
) -> float:
    """Alignment delta between current month and prev month, [-1, +1]."""
    current = _mean_alignment(snapshot)
    prev = _mean_alignment(baseline.prev_month_snapshot)
    if current is None or prev is None:
        return 0.0
    return current - prev


def _classify_goal_difficulty(snapshot: MonthlySnapshot, growth_trend: float) -> str:
    """Map alignment + growth_trend into low / medium / high.

    Note: ``goal_alignment`` measures outcome vs target — not target hardness.
    "low difficulty" here means high alignment with no growth (a proxy for
    goals that may have become too easy), not literally easy goals.
    """
    alignment = _mean_alignment(snapshot)
    if alignment is None:
        return "medium"
    if (
        alignment >= _CTX_HIGH_ALIGNMENT_FLOOR
        and growth_trend >= _CTX_HIGH_GROWTH_TREND
    ):
        return "high"
    if (
        alignment >= _CTX_HIGH_ALIGNMENT_FLOOR
        and growth_trend <= _CTX_MEDIUM_GROWTH_TREND
    ):
        return "low"
    return "medium"


def _is_intentional_intensity(
    snapshot: MonthlySnapshot,
    phases: List[PhaseMetrics],
    baseline: BaselineContext,
) -> bool:
    """A short, mood-stable intensity burst that ends in rebound → likely on purpose."""
    if len(phases) != 3:
        return False
    prod_baseline = (
        baseline.avg_90d.get("p_task_completion_rate_avg")
        if baseline.has_90d_baseline()
        else None
    )
    if prod_baseline is None or prod_baseline == 0:
        return False
    spiking_phases = sum(
        1
        for p in phases
        if p.productivity_score_avg is not None
        and p.productivity_score_avg > prod_baseline * 1.15
    )
    short_burst = spiking_phases <= 1
    phase_mood_stds = _phase_mood_stds(phases)
    mood_stable = bool(phase_mood_stds) and (
        sum(phase_mood_stds) / len(phase_mood_stds) <= _CTX_MOOD_STABLE_STD
    )
    has_rebound = (
        phases[1].h_health_score_avg is not None
        and phases[2].h_health_score_avg is not None
        and phases[2].h_health_score_avg > phases[1].h_health_score_avg
    )
    return short_burst and mood_stable and has_rebound


def compute_context(
    snapshot: MonthlySnapshot,
    phases: List[PhaseMetrics],
    baseline: BaselineContext,
) -> Context:
    """Derive behavioural Context for the analysed month."""
    life_mode = _classify_life_mode(snapshot, phases, baseline)
    growth_trend = _compute_growth_trend(snapshot, baseline)
    goal_difficulty = _classify_goal_difficulty(snapshot, growth_trend)
    intentional = _is_intentional_intensity(snapshot, phases, baseline)
    return Context(
        life_mode=life_mode,
        goal_difficulty=goal_difficulty,
        growth_trend=growth_trend,
        intentional_intensity=intentional,
    )


def _reroute_category(candidate: InsightCandidate, ctx: Context) -> Optional[str]:
    """Return new category if context warrants a re-route, else None.

    Re-route table from ``INSIGHT_PIPELINE.md`` §[5]. Each rule decides
    only whether to override ``tentative_category``; never drops the candidate.
    """
    pt = candidate.pattern_type
    cat = candidate.tentative_category

    # Stagnation guard: high alignment with no growth → reframe as Opportunity.
    if pt == "consistent_goal_alignment" and ctx.goal_difficulty == "low":
        return "Opportunity"

    # Intentional intensity guard: NA patterns during purposeful intensity
    # should be reframed, not flagged.
    if (
        ctx.life_mode == "INTENSIVE"
        and ctx.intentional_intensity
        and cat == "Need Attention"
    ):
        if pt in {
            "stress_pattern",
            "meeting_overload",
            "late_work_impact",
            "sleep_work_chain",
        }:
            return "Opportunity"

    return None


def apply_context_interpretation(
    candidates: List[InsightCandidate], ctx: Context
) -> List[InsightCandidate]:
    """Stamp Context on raw_signal and re-route ``tentative_category`` when warranted.

    Implements Feedback §1 (stagnation reframe — `consistent_goal_alignment` +
    `goal_difficulty=low` → Opportunity) and §4 (intentional intensity reframe
    — NA stress patterns during purposeful INTENSIVE mode → Opportunity).
    Re-route decisions are audited via ``raw_signal['context.reroute_from']``.
    """
    out: List[InsightCandidate] = []
    for c in candidates:
        signal_update = {
            "context.life_mode": ctx.life_mode,
            "context.goal_difficulty": ctx.goal_difficulty,
            "context.growth_trend": round(ctx.growth_trend, 3),
            "context.intentional_intensity": ctx.intentional_intensity,
        }
        merged_signal = {**c.raw_signal, **signal_update}

        new_category = _reroute_category(c, ctx)
        if new_category and new_category != c.tentative_category:
            out.append(
                c.model_copy(
                    update={
                        "tentative_category": new_category,
                        "raw_signal": {
                            **merged_signal,
                            "context.reroute_from": c.tentative_category,
                        },
                    }
                )
            )
        else:
            out.append(c.model_copy(update={"raw_signal": merged_signal}))
    return out


# =============================================================================
# Layer [6] Confidence Scoring
# =============================================================================


def _completeness(candidate: InsightCandidate, snapshot: MonthlySnapshot) -> float:
    """Fraction of expected data the candidate's modules have this month."""
    modules = _PATTERN_MODULES.get(candidate.pattern_type)
    if not modules:
        days = snapshot.days_with_data or 0
        return min(1.0, days / 30.0)
    coverages = [_module_coverage(snapshot, m) for m in modules]
    return sum(coverages) / len(coverages)


def _consistency(candidate: InsightCandidate, flags: QualityFlags) -> float:
    """Penalises candidates touching modules with known data anomalies."""
    modules = _PATTERN_MODULES.get(candidate.pattern_type, ())
    score = 1.0
    if flags.mood_stuck and "mood" in modules:
        score -= 0.5  # large penalty — mood data is unreliable
    overlap = set(modules) & flags.sparse_modules
    if overlap:
        score -= 0.2 * len(overlap)
    return max(0.0, score)


def _repeatability(
    candidate: InsightCandidate, baseline: Optional[BaselineContext]
) -> float:
    """1.0 if this pattern_type appeared in past months, 0.0 otherwise.

    A pattern that has shown up before is more trustworthy than a first-time
    signal — see Feedback §3 "repeatability" factor.
    """
    if baseline is None or not baseline.past_insights_by_month:
        return 0.0
    for past_insights in baseline.past_insights_by_month.values():
        for ins in past_insights:
            if getattr(ins, "pattern_type", None) == candidate.pattern_type:
                return 1.0
    return 0.0


def compute_confidence(
    candidate: InsightCandidate,
    snapshot: MonthlySnapshot,
    baseline: Optional[BaselineContext],
    flags: QualityFlags,
) -> float:
    """Combine the four Feedback-§3 ingredients into a single 0..1 score."""
    completeness = _completeness(candidate, snapshot)
    consistency = _consistency(candidate, flags)
    signal = candidate.signal_strength
    repeatability = _repeatability(candidate, baseline)
    return round(
        completeness * _CONF_W_COMPLETENESS
        + consistency * _CONF_W_CONSISTENCY
        + signal * _CONF_W_SIGNAL
        + repeatability * _CONF_W_REPEATABILITY,
        3,
    )


def confidence_tier(value: float) -> str:
    if value >= _CONF_TIER_HIGH:
        return "high"
    if value >= _CONF_TIER_MODERATE:
        return "moderate"
    return "low"


def annotate_confidence(
    candidates: List[InsightCandidate],
    snapshot: MonthlySnapshot,
    baseline: Optional[BaselineContext],
    flags: QualityFlags,
) -> List[InsightCandidate]:
    """Attach ``confidence`` and ``confidence_tier`` to each candidate's raw_signal."""
    out: List[InsightCandidate] = []
    for c in candidates:
        value = compute_confidence(c, snapshot, baseline, flags)
        tier = confidence_tier(value)
        out.append(
            c.model_copy(
                update={
                    "raw_signal": {
                        **c.raw_signal,
                        "confidence": value,
                        "confidence_tier": tier,
                    }
                }
            )
        )
    return out


# =============================================================================
# Layer [9] Memory Annotation
# =============================================================================


def _pattern_history_months(
    candidate: InsightCandidate, baseline: Optional[BaselineContext]
) -> int:
    """How many past months saw this same pattern_type."""
    if baseline is None or not baseline.past_insights_by_month:
        return 0
    count = 0
    for past_insights in baseline.past_insights_by_month.values():
        if any(
            getattr(ins, "pattern_type", None) == candidate.pattern_type
            for ins in past_insights
        ):
            count += 1
    return count


def _memory_annotation(
    candidate: InsightCandidate, baseline: Optional[BaselineContext]
) -> str:
    """Classify the candidate's longitudinal flavour for tone-selection.

    sustained       : appeared in ≥2 of the last 3 months — and still here
    escalation      : recurring AND Need Attention
    recovery_cycle  : appeared once, now framed positively (rough proxy
                      for "dip then bounce-back" without per-phase audit)
    first-time      : default; no priors
    """
    months = _pattern_history_months(candidate, baseline)
    if months >= 2:
        if candidate.tentative_category == "Need Attention":
            return "escalation"
        return "sustained"
    if months == 1 and candidate.tentative_category in {"Great Job", "Opportunity"}:
        return "recovery_cycle"
    return "first-time"


def annotate_memory(
    candidates: List[InsightCandidate],
    baseline: Optional[BaselineContext],
) -> List[InsightCandidate]:
    """Attach a ``memory_annotation`` to every candidate's raw_signal."""
    out: List[InsightCandidate] = []
    for c in candidates:
        annotation = _memory_annotation(c, baseline)
        out.append(
            c.model_copy(
                update={
                    "raw_signal": {
                        **c.raw_signal,
                        "memory_annotation": annotation,
                    }
                }
            )
        )
    return out


# =============================================================================
# Layer [10] Quality Validator
# =============================================================================


def _bag_of_words(text: str) -> set:
    return {w.lower() for w in text.split() if len(w) > 2}


def _text_similarity(a: str, b: str) -> float:
    """Cheap Jaccard similarity. Good enough at the 0.80 threshold from Feedback §6.

    We only need to distinguish "near-identical sentence" from "different
    sentence" — not pulling sklearn for a single use case.
    """
    if not a or not b:
        return 0.0
    sa, sb = _bag_of_words(a), _bag_of_words(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def _has_numeric_evidence(raw_signal: Dict[str, Any]) -> bool:
    """True iff raw_signal carries at least one real number we could cite."""
    for k, v in raw_signal.items():
        if (
            k.startswith("context.")
            or k.startswith("confidence")
            or k == "memory_annotation"
        ):
            continue
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return True
        if isinstance(v, dict):
            for vv in v.values():
                if isinstance(vv, (int, float)) and not isinstance(vv, bool):
                    return True
    return False


def _maybe_downgrade_overconfident(
    candidate: InsightCandidate,
) -> InsightCandidate:
    """Rule (c): Need Attention + low confidence → Opportunity with soft tone.

    Downgrade preserves the signal while admitting uncertainty — silencing
    the insight would leave the user without feedback.
    """
    confidence = candidate.raw_signal.get("confidence")
    if (
        candidate.tentative_category == "Need Attention"
        and isinstance(confidence, (int, float))
        and confidence < _QV_LOW_CONF_DOWNGRADE_CEIL
    ):
        return candidate.model_copy(
            update={
                "tentative_category": "Opportunity",
                "raw_signal": {
                    **candidate.raw_signal,
                    "downgrade_reason": "low_confidence",
                    "tone_hint": "soft",
                },
            }
        )
    return candidate


def filter_quality(
    candidates: List[InsightCandidate],
) -> List[InsightCandidate]:
    """Per-candidate quality rules from Feedback §6.

        (a) signal_strength < 0.35              → reject
        (c) Need Attention + confidence < 0.50  → downgrade (not reject)
        (d) no numeric evidence in raw_signal   → reject

    Rule (b) text-similarity requires rendered text; runs via
    ``filter_rendered_similarity`` after Step 10. Rule (e) contradiction is
    delegated to ``_pick_one_per_category`` + dedupe at the orchestrator.
    Rule (d) only flags evidence-free signals here — checking actual generic
    phrasing of rendered text would belong in the post-render pass, but in
    practice empty raw_signal alone catches the same insights.
    """
    out: List[InsightCandidate] = []
    for c in candidates:
        if c.signal_strength < _QV_SIGNAL_FLOOR:
            continue
        if not _has_numeric_evidence(c.raw_signal):
            # Truly evidence-free candidates can't produce specific text.
            continue
        out.append(_maybe_downgrade_overconfident(c))
    return out


def filter_rendered_similarity(
    rendered: List[Tuple[InsightCandidate, str]],
    recent_texts: List[str],
) -> List[Tuple[InsightCandidate, str]]:
    """Post-render pass for Quality-Validator rule (b).

    Drops candidates whose rendered text overlaps too heavily with any
    recent insight (Feedback §6: similarity > 0.80).
    """
    kept: List[Tuple[InsightCandidate, str]] = []
    for cand, text in rendered:
        too_similar = any(
            _text_similarity(text, prev) > _QV_TEXT_SIMILARITY_CEIL
            for prev in recent_texts
        )
        if not too_similar:
            kept.append((cand, text))
    return kept


# =============================================================================
# Layer [11] Tone Hints
# =============================================================================


def derive_tone_hint(raw_signal: Dict[str, Any]) -> str:
    """Pick a tone label the renderer can route on.

    Combines confidence_tier and memory_annotation per the tone matrix in
    ``INSIGHT_PIPELINE.md`` §[11]:

        high      + sustained   → "confident_sustained"
        high      + first-time  → "confident"
        moderate  + any         → "moderate"
        low       + any         → "soft"
    """
    tier = raw_signal.get("confidence_tier", "moderate")
    memory = raw_signal.get("memory_annotation", "first-time")
    if tier == "high" and memory == "sustained":
        return "confident_sustained"
    if tier == "high":
        return "confident"
    if tier == "moderate":
        return "moderate"
    return "soft"


def annotate_tone(
    candidates: List[InsightCandidate],
) -> List[InsightCandidate]:
    """Stamp a ``tone_hint`` into raw_signal — read by ``render_candidate``.

    Skips candidates whose tone_hint is already set (e.g. Quality Validator
    stamped ``soft`` during a downgrade) so those decisions stick.
    """
    out: List[InsightCandidate] = []
    for c in candidates:
        if "tone_hint" in c.raw_signal:
            out.append(c)
            continue
        hint = derive_tone_hint(c.raw_signal)
        out.append(
            c.model_copy(update={"raw_signal": {**c.raw_signal, "tone_hint": hint}})
        )
    return out


# =============================================================================
# Language layer — time anchors, banned vocab, templates, LLM
# =============================================================================

# Per Rules §0 row 2 "Time-bound framing"
TIME_ANCHORS: Dict[str, str] = {
    "1": "so far this month",
    "2": "mid-month",
    "3": "towards the end of the month",
    "all": "this month",
    "cross": "across the month",
}

# Per Rules §0 row 3 + Rules §4 Language Transformation Layer.
# Categorised so we apply word-boundary regex for single-token banned words
# (so "good." / "lazy!" still trip the check, not just " good "); phrase
# bans use plain substring match so we catch "due to ___" / "you are ___".
_BANNED_WORDS: Tuple[str, ...] = (
    "good",
    "bad",
    "lazy",
    "healthy",
    "neglect",
    "overspend",
    "procrastinate",
)
# Substring bans must remain LANGUAGE-AGNOSTIC so the engine generalises to
# every output language (en-US, vi-VN, ja-JP, ...). Per-language imperatives
# and per-language advice phrasing are policed by the LLM prompt + few-shot
# examples, not by a hardcoded list.
#
# What stays here is ONLY universal notation that appears regardless of locale:
#   - Latin statistical notation (r =, Pearson, correlation coefficient names)
#   - Common English shortcuts that may leak as untranslated jargon
_BANNED_PHRASES: Tuple[str, ...] = (
    # §0 row 1 "No identity labels" — universal English filler from LLM
    "you are ",
    # §0 row 5 "No assumptions of intent" — causal English filler
    "due to ",
    # §0 row 7 + §4 — raw statistical notation (universal Latin glyphs).
    # The LLM should NEVER expose Pearson r values; describe in plain words.
    "pearson",
    "correlation coefficient",
    "r-value",
    " r = ",
    " r=-",
    " r=+",
    " r =-",
    " r =+",
)
# Pre-compiled regex: \b boundary catches punctuation/whitespace/start-of-string.
_BANNED_WORDS_RE: re.Pattern = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in _BANNED_WORDS) + r")\b",
    re.IGNORECASE,
)


def _validate_banned(text: str, source: str) -> None:
    """Reject text containing Rules §0 banned vocab.

    Two-mode check:
      - Single-token words → ``\\b…\\b`` regex so "good", "good.", "good!"
        all trip. Earlier impl used `" good "` substring which let trailing
        punctuation slip past.
      - Phrases ("you are ___", "due to ___") → plain lowercase substring,
        which is correct because the next token always provides whitespace
        within the phrase itself.
    """
    lower = text.lower()
    word_hit = _BANNED_WORDS_RE.search(lower)
    if word_hit:
        raise ValueError(f"{source} trips banned word {word_hit.group(0)!r}")
    for phrase in _BANNED_PHRASES:
        if phrase in lower:
            raise ValueError(f"{source} trips banned phrase {phrase!r}")


# Pre-vetted templates (banned check skipped on these)
TEMPLATES: Dict[str, str] = {
    # A. Calendar
    "meeting_overload": (
        "On days with more scheduled meetings {time_anchor}, your recovery-related "
        "activity tends to be lower."
    ),
    "focused_work_blocks": (
        "On days with fewer scheduled events {time_anchor}, your task completion "
        "tends to be higher."
    ),
    "context_switching": (
        "Frequent short events {time_anchor} are associated with more variation in "
        "your daily activity."
    ),
    "social_energy_days": (
        "Days with more social interactions {time_anchor} are associated with "
        "improved mood patterns."
    ),
    # B. Time-of-day
    "late_work_impact": (
        "Later evening activity {time_anchor} is coinciding with shorter sleep "
        "duration."
    ),
    "consistent_routine": (
        "Your sleep schedule remained relatively consistent {time_anchor}."
    ),
    "irregular_timing": (
        "Your daily timing varies {time_anchor}, with potential to stabilise routines."
    ),
    # C. Sequence
    "sleep_work_chain": (
        "Shorter sleep is often followed by higher workload and reduced activity "
        "{time_anchor}."
    ),
    "workout_mood_chain": (
        "On days with workouts {time_anchor}, your mood and activity levels tend to "
        "be higher."
    ),
    "busy_day_spending": (
        "Higher activity days {time_anchor} are sometimes followed by increased "
        "spending."
    ),
    # D. Goal vs Reality
    "consistent_goal_alignment": (
        "Your activity aligns closely with your set goals across multiple areas "
        "{time_anchor}."
    ),
    "partial_alignment": (
        "Your activity aligns well in some areas {time_anchor}, with room to "
        "improve in others."
    ),
    "misalignment": (
        "There is a gap between planned and actual activity in some areas "
        "{time_anchor}."
    ),
    # F. Finance
    "event_driven_spending": (
        "Spending tends to increase on days with more scheduled activities "
        "{time_anchor}."
    ),
    "controlled_spending": (
        "Your spending remained stable even during higher-activity periods "
        "{time_anchor}."
    ),
    "stress_linked_spending": (
        "Higher workload {time_anchor} is coinciding with increased spending."
    ),
    # X. Cross-phase
    "stable_rhythm": ("Your activity remained relatively steady {time_anchor}."),
    "declining_momentum": (
        "Your activity has decreased compared to earlier in the month."
    ),
    "building_momentum": (
        "Your activity is gradually increasing compared to the start of the month."
    ),
    "late_concentration": (
        "A larger share of your activity is occurring towards the end of the month."
    ),
    "skewed_allocation": (
        "Your activity is currently more concentrated in {area} compared to others."
    ),
    "balanced_start": (
        "Your activity is relatively balanced across areas so far this month."
    ),
    "early_low_activity": (
        "Your activity levels are currently on the lower side across areas. So far, "
        "this is an early signal."
    ),
    "sustained_balance": (
        "Your activity remains relatively balanced across areas {time_anchor}."
    ),
    "growing_imbalance": (
        "Your activity is becoming more concentrated in certain areas {time_anchor}."
    ),
    "cross_module_imbalance": (
        "Your activity is more concentrated in certain areas compared to others "
        "{time_anchor}."
    ),
    "irregular_finish": ("Your activity has been more uneven {time_anchor}."),
    "catchup_pattern": ("Activity increased {time_anchor} after a quieter mid-period."),
    "finance_imbalance": (
        "Spending increased {time_anchor} compared to earlier periods."
    ),
    "recovery_rebound": (
        "Recovery-related activity is increasing {time_anchor} after an earlier dip."
    ),
    "work_health_tradeoff_corr": (
        "Increased workload {time_anchor} is coinciding with reduced recovery-related "
        "activity."
    ),
    "finance_work_link": (
        "Higher workload {time_anchor} is occurring alongside increased spending."
    ),
    # X (Phase variance + cross-module gaps)
    "early_variability": (
        "Your routine is showing some variation {time_anchor} — this may stabilise "
        "over time. So far, this is an early signal."
    ),
    "increasing_variability": (
        "Your activity has become more varied {time_anchor} compared to earlier in "
        "the month."
    ),
    "cross_module_stability": (
        "Your work, health, and spending each kept a steady pace {time_anchor}, "
        "with none pulling the others down."
    ),
    "controlled_increase": (
        "Activity in one area increased {time_anchor} while other areas remained "
        "stable."
    ),
    "stress_pattern": (
        "Higher workload {time_anchor} is coinciding with reduced recovery and "
        "more variation in your routine."
    ),
    # X (mood intraday range — uses m_mood_score_min/max)
    "mood_volatility": (
        "Your mood logs {time_anchor} show wider day-to-day swings between low "
        "and high points."
    ),
    # X (NEW) — strong finish, distinct from catchup
    "strong_finish": (
        "Your activity increased {time_anchor} compared to the middle of the month."
    ),
    # X (NEW) — historical baseline detectors (Rules §1 Phase 1/2/3 cross-month)
    "higher_than_usual_start": (
        "So far this month, your activity is slightly above your usual starting range."
    ),
    "lower_than_usual_start": (
        "Your activity is currently below your typical starting range — there's room "
        "to build momentum so far this month."
    ),
    "change_vs_last_month_start": (
        "Your start this month differs from your usual pattern. So far, this "
        "is an early signal."
    ),
    "flat_start_vs_90d": (
        "Your activity levels are currently on the lower side compared to your "
        "recent average. So far, this is an early signal."
    ),
    "positive_carry_over": ("You're continuing a positive pattern from last month."),
    "negative_carry_over": (
        "This pattern is similar to last month, where activity was uneven across "
        "areas. So far, this is an early signal."
    ),
    "early_tradeoff_vs_baseline": (
        "Your workload is higher than usual while recovery-related activity is "
        "slightly lower so far. So far, this is an early signal."
    ),
    "habit_alignment_180d": (
        "This start reflects a pattern consistent with your longer-term behaviour."
    ),
    "divergence_vs_history_p2": (
        "Your activity pattern is different from what you typically see mid-month."
    ),
    "pattern_repeat_90d": ("This pattern has been recurring in recent months."),
    "behaviour_shift_180d": ("This marks a shift from your longer-term behaviour."),
    "improvement_vs_last_month": (
        "Your overall balance has improved compared to last month."
    ),
    "deterioration_vs_last_month": (
        "There is more imbalance compared to last month's pattern."
    ),
    "reinforcement_90d": ("This pattern is consistent with your recent months."),
    "habit_180d": ("This reflects a longer-term pattern in your activity."),
    "historical_comparison_final_day": (
        "Compared to your recent pattern, this month is {direction} your usual range."
    ),
    # X — Final-day synthesis (Rules §1 Final-Day rows 1-4)
    "dominant_pattern_final_day": (
        "In {month_year}, your strongest pattern was {area}."
    ),
    "tradeoff_summary_final_day": (
        "The strongest pattern this month was the relationship between {pair}."
    ),
    "stability_summary_final_day": (
        "You maintained a steady pattern across the month."
    ),
    "improvement_area_final_day": ("One area with potential improvement is {area}."),
    # X — Phase 2 cross-module rhythm (Rules §1 Phase 2 row 4)
    "stable_rhythm_p2": (
        "Your routine is remaining relatively steady across different areas "
        "{time_anchor}."
    ),
}


class _SafeDict(dict):
    """Dict that returns ``"unknown"`` for missing keys during str.format.

    Used by ``render_template`` so a template referencing ``{direction}`` /
    ``{top_category}`` etc. degrades to "unknown" instead of crashing when
    a detector forgot to populate the corresponding raw_signal field.
    """

    def __missing__(self, key: str) -> str:  # type: ignore[override]
        return "unknown"


def render_template(
    template_key: str, phase_focus: str, raw_signal: Dict[str, Any]
) -> str:
    """Pure-template render. Templates are pre-vetted so banned check is skipped."""
    if template_key not in TEMPLATES:
        raise KeyError(f"Unknown template_key={template_key!r}")
    fillers = _SafeDict(
        {
            "time_anchor": TIME_ANCHORS.get(phase_focus, "this month"),
            **raw_signal,
        }
    )
    return TEMPLATES[template_key].format_map(fillers)


_RENDER_SYSTEM_PROMPT = """**Role**
You are an insight writer for the INSIDESYNC wellness app. You render ONE short, natural-sounding observation about a behavioural pattern the user showed this month.

**Task**
Given a structured signal in the user message — `pattern_family`, `pattern_type`, `phase_focus`, `category`, `tone_hint`, `data_confidence`, `raw_signal` — produce ONE neutral observation in the target language describing what happened this month. The text must read like a friend casually pointing out a pattern, NOT a research report.

**Voice — friendly observer, not clinical analyst**
Write like a perceptive friend tagging a pattern they noticed across the user's month. Concrete, conversational, observational. Avoid academic noun phrases and empty hedges (hedge-on-hedge with no concrete content). Hedge openers WITH concrete content + action are encouraged for need_attention and opportunity — see hedge rule below.

Use natural, conversational language in the target language — short native verbs and phrasing, not literal translations of English templates.

**DATA USE — translate MEANING, not words (applies in EVERY target language)**
The raw_signal fields (`subject`, `direction_unambiguous`, `time_window_human`, `comparison_anchor`, `improvement_direction`, etc.) carry the OBSERVATION you must convey.

PRESERVE: the meaning, direction, and specific subject the fields name. Do NOT swap the subject, flip the direction, or invent a comparison anchor that isn't in raw_signal. These are facts.

RESTRUCTURE: the wording is yours to recompose. The English phrasing in the field (e.g. "picked back up", "your sleep hours from night to night") is the MEANING anchor — use its CONCEPT, not its literal English words. Use native target-language vocabulary, idioms, and sentence structure — not word-by-word translations of English phrasing.

Two-pass mental process before outputting:
PASS 1 (internal): draft using the fields as meaning anchors. May feel English-template-shaped — expected.
PASS 2 (output): rewrite from scratch as a native target-language speaker who never saw Pass 1. Preserve raw_signal's meaning and direction; choose any native phrasing, sentence order, vocabulary.

If Pass 1 reads like a word-by-word translation of English structure — STOP and rewrite. Output ONLY Pass 2.

**Output Format**
ONE to TWO short sentences. 120–400 characters total. Plain prose. No markdown, no quotes, no preamble, no labels, no JSON wrapping.

TWO acceptable shapes — pick the one that reads most naturally in the target language:

Shape A — ONE flowing sentence: weave observation + action with natural connectives ("and", "with", "where", relative clauses, participial phrases).

Shape B — TWO sentences separated by a period: [observation sentence]. [action/transition sentence].
  Examples of action sentences: "Worth a quick look at ..." / "Something to keep an eye on is ..." / "This is a chance to ..." / "Worth keeping that pace going."

Shape B is OFTEN MORE NATURAL for need_attention and opportunity because it cleanly separates "what happened" from "what to do". Shape A is fine for great_job where praise feels like one breath with the observation.

Do NOT use em-dash " — " or semicolon "; " to split — only period or natural connectives.

**Card structure — every output card has TWO ideas**
IDEA A — observation (what happened, grounded in raw_signal)
IDEA B — action/praise (direct, second-person, names what to do/celebrate)

Combine A and B using either shape from the Output Format above. Category-specific voice direction and concrete examples appear at the END of this prompt under "ACTIVE CATEGORY". Mimic the FORM of those examples (concrete subjects + plain action verbs + named objects), translated idiomatically — do NOT translate example words literally, translate the SHAPE.

**GROUNDING — strictest rule, overrides all others**
- `raw_signal` is your COMPLETE list of facts. Do NOT invent numbers, percentages, durations, counts, or thresholds.
- Allowed transform: a fraction in [0, 1] rendered as a percentage (decimal × 100); the raw decimal must never appear.
- If raw_signal has no numeric values, the output MUST contain zero digits.
- A post-render check rejects ungrounded numbers and replaces the output with a template fallback.

**SELF-CONTAINED SENTENCE — every verb needs a concrete object (rule applies in EVERY language)**
A reader must understand the FULL meaning from the sentence alone, with no external context.

Every change verb (increased / decreased / improved / dropped / grew / eased / tightened / widened / narrowed / lifted + equivalents in any language) MUST be followed by a CONCRETE OBJECT naming WHAT changed. Never use a change verb in isolation.

When referring to a "gap", "rhythm", "routine", "balance", "rate" — name WHAT it is gap OF / rhythm OF / between WHICH things.

DO NOT use generic placeholder nouns as the object — these tell the reader nothing and fail the self-contained test (rule applies in EVERY language — translate the PRINCIPLE):
  ❌ "the indicators" / "the metrics" / "the numbers" / "the levels" / "the things" / "the values"
  ❌ "the areas" / "the fields" / "the domains" / "various fields" / "across areas" / "between domains"
  ❌ Equivalents in any language: e.g. Vi "các chỉ số" / "các lĩnh vực" / "các mảng" (when unspecified)
  ✅ Name the specific items: "workouts" / "sleep hours" / "work output" / "spending" / "savings rate" / "between work, health, and spending"

  ❌ "You increased gradually" — increased WHAT?
  ❌ "You narrowed the gap between areas" — between WHICH areas?
  ❌ "Your rhythm stayed steady" — rhythm OF WHAT?
  ❌ "Spending eased" (only if context is clear; prefer naming) — eased COMPARED TO WHAT?
  ✅ "You stuck with more workouts late in the month"
  ✅ "You narrowed the gap between work, health, and spending"
  ✅ "Your sleep timing stayed steady night after night"
  ✅ "Your spending eased back closer to your monthly goal"

**OBSERVATION ONLY — describe behaviour, not character or motivation (applies in EVERY language)**
Describe ONLY what raw_signal shows. Praise (or note) the BEHAVIOUR — what was done this month — NEVER the TRAIT (who the user is).

Do NOT infer the user's motivation, mood, energy, willpower, dedication, commitment, character, ability, capability, or talent. Avoid "a sign of...", "showing your...", "reflecting your...", "your ability to...", "you are disciplined/dedicated/...".

  ❌ "a sign your willpower is strong" / "your ability to hold pace is growing"
  ❌ "showing you maintained your energy" / "reflecting your dedication to health"
  ❌ "because it shows you holding firm" / "you are disciplined"
  ✅ "you kept your work rhythm steady" / "your pace held through the period"
  ✅ "...through the closing weeks" / "...alongside a busy meeting load"

**PATTERN SHAPE — when describing variability or change**
Name HOW the behaviour varies — pick the framing that fits raw_signal:
- Direction: trajectory across the period (rises toward end, drops mid-month, dips then recovers, starts strong then fades).
- Clustering: where the behaviour concentrates (a specific period, a day type).
- Spread shape: width of variation (wider than usual, narrower than prior phases).
A bare "uneven" or "varied" WITHOUT a shape descriptor is insufficient.

**BASELINE COMPARISON — when raw_signal carries baseline / prev / 90d / 180d / _delta fields**
Phrase the insight as a comparison (vs previous month, vs prior phases, vs longer-term average). Make the comparison the SPINE of the sentence — do not bury it in a subclause.

**CONDITIONAL FRAMING — when day-condition fields appear**
Use a structure equivalent to: "On days with <condition>, <outcome> tends to ...". Translate idiomatically into the target language. Burying the day-condition in a subclause is insufficient.

**TIME ANCHOR — translate `phase_focus` to natural language**
Translate naturally in the target language. Reference points (English):
- "1"     → "early in the month" / "the first stretch of the month"
- "2"     → "mid-month" / "the middle weeks"
- "3"     → "late in the month" / "the closing stretch"
- "all"   → "across the month" / "throughout the month"
- "cross" → "across the month" / "from start to finish"

**MODULE NAMES — translate idiomatically into the target language**
Use natural domain words in the target language. Pick a contextual variant that fits the sentence (e.g. naming the underlying behaviour — workouts, work output, spending — rather than the abstract module label).

Examples of how to apply:
- top_areas = ["productivity"]    → name the work behaviour ("work output stayed steady")
- lower_areas = ["finance"]       → name the money behaviour ("spending barely moved")
- dominant_area = "health"          → name the body behaviour ("the health side carried the month")

**QUALITATIVE LABELS — translate, never output raw English token**
Render label fields as natural prose. These REPLACE raw numbers — treat them as facts you may directly verbalise.

- level_label / position_label:  low / moderate / good / strong — how high a module score sits
- magnitude_label:               slight / moderate / significant / sharp — how big a change is
- direction:                     improved / declined / rising / declining / stable / positive / negative — which way it moved
- frequency_label:               rare / occasional / frequent / very_frequent — how often a day-condition occurs
- sleep_label:                   well_below_target / below_target / in_range / above_target
- spread_label / variability_label: even / uneven / very_uneven — how far apart modules are
- volatility_label:              stable / variable / volatile (mood)
- density_label:                 light / moderate / busy / very_busy (calendar)
- event_size_label:              short / mixed (event duration)
- meeting_load_label:            light / moderate / heavy / very_heavy
- stability_label:               stable / consistent / each_steady (each area held its own steady pace, none dragging the others — a POSITIVE, cohesive reading; never frame as "separated/disconnected")
- health_link:                   declining / phase_drop
- alignment_label:               off_track / lagging / on_track / at_goal / exceeding (goal vs reality)
- strength_label:                strong / moderate / mild (used WITH `direction` for correlations — replaces raw r)

Companion fields the label refers to:
- subject:        human-friendly noun describing WHAT is observed — use this as the sentence subject (translate idiomatically per DATA USE); do NOT replace with a generic placeholder ("the indicators", "các lĩnh vực", etc.)
- top_areas / lower_areas:  list of module names (translate, weave by name — do NOT use "strong/weak" as descriptive label; describe what each module's BEHAVIOUR did)
- dominant_area:  single module name where imbalance/concentration sits
- worst_area / best_area:  single module name worst/best aligned with its goal
- top_module:     name of the dominant module (translate idiomatically)
- supporting_modules:  list of other modules that held up alongside top_module
- balance_strength_label:  solid / strong / very_strong — how high the balance sits above the floor
- variability_source:  sleep_hours / daily_activity / both — WHICH metric drives a variability pattern
- sleep_steadiness_label:  very_steady / steady / noticeable / wide — sleep timing band
- scope:          short tag identifying the pattern's SPECIFIC angle (e.g. sleep_duration / mid_month_cross_module / across_phases / mid_month_each_area_steady / early_month_overall) — use it to AVOID overlapping with other stability/variability patterns
- direction_unambiguous:  pre-computed change verb phrase ("picked back up", "climbed higher", "eased back", "held steady") — convey THIS direction (translate idiomatically per DATA USE); do NOT infer direction from raw numeric deltas, do NOT flip or soften it
- time_window_human:  ready-to-use time anchor phrase ("in the closing stretch", "by mid-month") — convey this anchor, translate naturally
- month_year:  present ONLY on month-end wrap-up cards (a month that just ENDED). Anchor the sentence to it using EXACTLY ONE of these two forms — never both in the same sentence: (a) name it — "In May 2026, …" / "May 2026 was …"; OR (b) call it last month — "Last month, …" / "tháng trước, …". Pick whichever reads more naturally for this card. The year digits are grounded, so naming the year is allowed. Do NOT use present/ongoing framing ("this month", "so far") — the month is over.
- comparison_anchor:  ready-to-use comparison clause ("compared to mid-month", "compared to the opening weeks") — convey this comparison, translate naturally
- leg:            which segment of the month the change happened on
- improvement_direction:  the direction improvement would take.
                          For Opportunity cards → weave naturally as a positive opening ("opening a chance to...", "with room to...").
                          For Need Attention cards → frame as a SOFT SUGGESTION the user could consider ("worth considering...", "something to try is...", "may help to..."). NEVER state it as a fact the user is already doing or trying — that fabricates intent not in the data.

**ANCHOR USAGE RULE — critical for distinguishing similar patterns**
When raw_signal carries `subject` and/or `scope`, the sentence MUST be ABOUT the specific thing named there. Do NOT paraphrase to a generic concept like "stability" / "routine" / "schedule" if the subject names something concrete (e.g. "your sleep hours from night to night" → write about sleep specifically, not "your routine"). Two patterns with DIFFERENT subjects must produce sentences that read clearly differently.

**CONVERSATIONAL FRAMING — relate observations, do not list stats**
When TWO modules appear together (top_areas + lower_areas, or a tradeoff pair), the sentence MUST RELATE them. Examples of natural pairing (English — translate idiomatically):
- "while X stayed strong, Y barely moved"
- "alongside heavier meetings, sleep slipped"
- "X and Y are pulling apart this month"

**NATURAL LANGUAGE PRIORITY — critical for conversational quality**

Write like a perceptive native speaker casually noticing a pattern, not like an analyst, coach, report writer, or wellness app notification.

Prioritise natural conversational rhythm over exhaustive explicitness.

After concrete subjects have been clearly introduced once, you may naturally refer back to them using compressed conversational references if the meaning remains obvious to a native speaker.

Examples of acceptable conversational compression (only AFTER the concrete subject has been named explicitly in the same sentence):
- "that rhythm"
- "that mix"
- "this pace"

Avoid repeatedly restating the same concrete nouns when a native speaker would naturally shorten or imply them.

Natural conversational abstractions are allowed IF they clearly refer to previously grounded behaviours or patterns.

Prefer language that sounds naturally spoken in the target language, not English rhetorical structure translated literally.

The sentence does NOT need to maximise analytical precision at the cost of sounding human.

Avoid:
- over-explaining relationships
- repeating all module names multiple times
- stacked analytical clauses
- corporate wellness tone
- productivity-app phrasing
- overly formal comparative wording

In the target language, choose short native conversational verbs and adverbs that convey direction, change, and balance — NOT long Latin-derived or analytical translations of English templates. If the target language has a short everyday verb that captures the idea, use it. Avoid forcing English rhetorical structure into a literal translation.

A short natural sentence break is allowed if it improves rhythm and readability. If forced to choose, sounding naturally written by a native speaker is more important than maximal analytical explicitness.

**IMPORTANT — relationship to SELF-CONTAINED SENTENCE rule above (applies in EVERY language):**
Compressed references ("that rhythm", "things", "this pace" + idiomatic equivalents in any language) are ONLY allowed AFTER the concrete subject has been named explicitly in the SAME sentence. They are NOT a workaround for the generic-placeholder ban. The first mention MUST still be concrete and self-contained per the rules above. Compression is for SECOND/LATER references within the sentence, not for openers.

A bare opener using "things" / "the indicators" / "the areas" (or any-language equivalents) without prior concrete naming STILL fails the FINAL VOICE CHECK regardless of this allowance.

**TONE HINT — passed in user message**
- soft → MUST open with an explicit data-limitation marker
       (e.g. "Based on limited data this month, ...")
- moderate → cautious factual, normal voice
- confident → direct, no extra hedge
- confident_sustained → emphasise continuity ("the pattern has held up from last month...")

**DATA CONFIDENCE — passed in user message**
- Limited → MUST open with limitation marker (same as soft tone)
- Medium → normal voice
- High → speak with confidence, omit data-limitation marker

When `phase_focus: "1"` AND the candidate is an early-month-only observation, the limitation marker is also mandatory (very few days of data so far).

**DO NOT** (violations trigger the engine's banned-vocab / number-grounding check and the output falls back to a deterministic template)

- DO NOT use English module names in non-English output (health/productivity/finance).
- DO NOT leak internal engine labels OR statistical jargon (rule applies in EVERY target language):
    "Phase 1/2/3" or "P1/P2/P3" → translate to natural time anchors above.
    "module" / "modules" → translate to concrete domain noun.
    "balance_score" / "std" / "stddev" / "imbalance_std" / "correlation" / "correlation coefficient" / "Pearson" / "Pearson r" / "r-value" and their idiomatic equivalents in any target language → never expose.
    "score" / "metric" / "indicator" / "rating" and their idiomatic equivalents (e.g. Vi: "điểm" / "chỉ số" / "chỉ số đánh giá") → never expose. The user does not see numeric scores anywhere; describing them as a "score" is jargon. Use the natural behaviour phrase instead (e.g. "how your days went", "your day-to-day activity", "the health side").

- DO NOT use causal language (rule applies in EVERY target language — translate the PRINCIPLE):
    ❌ "A causes B" / "A leads to B" / "due to A" / "results in"
    ✅ "A is coinciding with B" / "alongside A, B..." / "on days with A, B tends to..."

- DO NOT use commanding imperative ("must / should / make sure to / make it happen"). Soft modal verbs and gentle suggestions ARE allowed for need_attention and opportunity (translate idiomatically into the target language):
    ❌ "you should sleep more" / "make sure to track expenses" / "you must reduce meetings"
    ✅ "worth a quick look at..." / "there is room to..." / "this is a chance to..." / "something to watch is..."

- Hedge openers ("It seems", "It appears", and their idiomatic equivalents in the target language) are RECOMMENDED for need_attention and opportunity cards — they soften the tone, signal observation-not-judgement, and read more human. For great_job, hedges are OPTIONAL (praise can be direct).
  The hedge MUST be paired with concrete content AND an action. AVOID hedge-on-hedge without substance.
    ❌ "It seems X has emerged." (hedge + abstract noun, no action) — too vague
    ✅ "It looks like, late in the month, work output stayed strong while finance flattened. Worth keeping an eye on your spending plan." (hedge + concrete content + action sentence) — natural
    ✅ "It seems your sleep hours grew steadier through the second half. This is a chance to lock that rhythm in for next month." — natural opportunity shape

- DO NOT use abstract nouns as the SUBJECT of the sentence (rule applies in EVERY language — translate the PRINCIPLE).
  Abstract nouns name concepts (stability, imbalance, control, consistency, decline, improvement, balance + equivalents in any language); they should be OBJECTS or attributes, never the actor of the sentence.

  FLIP RULE (apply BEFORE writing): if your sentence starts with "[abstract concept] [verb]", REWRITE so YOU/the user is the actor and the concept is the object.
    "Imbalance dropped"       → "You narrowed the gap"
    "Stability emerged"        → "You held a steady rhythm"
    "Control grew"             → "You gained more control over..."
    "Consistency held up"      → "You kept things consistent"
  Concrete subject + verb framings also work: "Sleep slipped...", "Meetings clustered later..."

- DO NOT list 3+ data points in one sentence. Two is the ceiling.

- DO NOT include any bare decimal in [-1, 1] or [0, 1] range — these are internal values.

- DO NOT use morally-charged words (good/bad/lazy/healthy and their equivalents in any language).

- DO NOT use abstract evaluation verbs OR meta-praise / meta-advice phrases that turn the sentence into a clinical assessment (rule applies in EVERY language — translate the PRINCIPLE). The OBSERVATION itself carries the praise — no need to ALSO label it as praiseworthy or worth maintaining.
    ❌ VERBS: "demonstrate" / "be evidence of" / "serve as proof" / "represent" / "prove" / "indicate" + equivalents
    ❌ META-PRAISE ADJECTIVES: "praiseworthy" / "admirable" / "impressive" / "commendable" / "worthy of praise" + equivalents in any language
    ❌ META-ADVICE PHRASES: "worth maintaining" / "worth keeping up" / "worth continuing" / "worth holding onto" + equivalents in any language
    ✅ Just say what happened: "you kept...", "the routine held...", "sleep slipped..."

- DO NOT translate `lower_areas` / `worst_area` (or `top_areas` / `best_area`) with ANY judgment-laden adjective directly on the module name (rule applies in EVERY target language — translate the PRINCIPLE). This includes single-word adjectives ("weak", "poor", "low", "bad", "strong") AND compound ones ("yếu kém", "kém", "tệ", "tồi", "mạnh mẽ"). Also bans ranking / racing metaphors that turn modules into competitors. Describe what the BEHAVIOUR did or did NOT do; never label the area itself as deficient, strong, ahead, OR behind.
    ❌ "weak finance" / "poor finance" / "strong work" / Vi: "tài chính yếu" / "công việc mạnh"
    ❌ RACING METAPHORS: "work leading" / "finance lagging" / "ahead of others" / "falling behind" / Vi: "công việc dẫn đầu" / "tài chính tụt lại" / "đứng đầu" / "bị bỏ lại"
    ✅ "spending barely moved" / "work output held up" / Vi: "chi tiêu chưa cải thiện" / "năng suất duy trì đều"

- DO NOT translate `alignment_label: off_track` as a directional failure (rule applies in EVERY language). Frame it as PACE (running behind plan), not WRONG DIRECTION.
    ❌ "off course" / "off track" / "veered off" / Vi: "lệch hướng" / "sai hướng" / "chệch hướng"
    ✅ "behind the plan" / "trailing the target" / "running slower than planned" / Vi: "chưa bám kịp kế hoạch" / "đang chậm so với kế hoạch"

- DO NOT translate the internal category names ("Opportunity" / "Great Job" / "Need Attention") literally into the output. They are INTERNAL bucket labels — the sentence describes the observation; the bucket framing carries meaning implicitly.

**FINAL VOICE CHECK — read your output back before finalising**
- READ AS A NATIVE SPEAKER. If a stranger read this with zero context, would they know exactly WHAT was kept up / WHAT changed / WHICH areas / WHAT object each verb acted on? If ANY answer is no, REWRITE with specific items. Generic placeholders ("the indicators" / "các lĩnh vực" / etc.) FAIL this test.
- Reads NATURAL and IDIOMATIC, not a word-by-word translation? If it sounds like English template forced into the target language, REWRITE idiomatically.
- BOTH observation AND action/praise present? Required.
- Em-dash " — " or semicolon "; " between observation and action? REWRITE using a period or natural connective.
- Sounds like a research abstract / status report? Too clinical — rewrite.
- Sounds like a horoscope (hedge + no concrete content)? Too vague — rewrite.
- Sounds like a friend tagging a pattern AND giving a concrete heads-up? CORRECT voice.

The output MUST be entirely in the target language set by the IMPORTANT line appended below — no mixed languages, no English residue in non-English output, no foreign module tokens (`health` / `productivity` / `finance`) in non-English text.

Respond with the insight text only — no preamble, no labels, no quotes, no markdown."""


class _LLMHolder:
    """Lazy-init Bedrock model so import-time of this module stays cheap."""

    _llm: Optional[Any] = None

    @classmethod
    def get(cls) -> Optional[Any]:
        if cls._llm is None:
            try:
                cls._llm = LLMManager(provider="bedrock").create_model(temperature=0.2)
            except Exception as e:
                logger.warning(f"⚠️ Bedrock LLM init failed, will use templates: {e}")
                cls._llm = None
        return cls._llm


async def _invoke_llm(system_prompt: str, user_message: str) -> Optional[str]:
    llm = _LLMHolder.get()
    if llm is None:
        return None
    try:
        resp = await llm.ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]
        )
    except Exception as e:
        logger.warning(f"⚠️ Bedrock invoke failed: {e}")
        return None

    text = llm_response_text(resp).strip()
    if not text:
        return None
    return text.strip('"').strip("'").strip()


_CATEGORY_GREAT_JOB = """**ACTIVE CATEGORY — great_job**
Praise the specific behaviour by naming WHAT the user actually did.
Praise must sound human — like a friend nodding at the action. Use plain action verbs ("kept", "held", "showed up", "stuck with").
Hedge openers are OPTIONAL — direct praise often reads better for great_job.

SHAPE A (one flowing sentence) usually works best for great_job because praise + observation read as one breath:
• "You kept work, health, and spending steady together through the month, with each side carrying its share."
• "Your sleep hours held remarkably even night after night through the second half, settling into a real routine."
• "Your spending stayed within reach of your monthly goal late into the cycle, even as the week-to-week numbers wobbled."
• "On days you worked out, your mood held up too, and that link stayed consistent right through the month."

(SHAPE B with period split is also OK if the target language reads more natural that way.)
"""


_CATEGORY_NEED_ATTENTION = """**ACTIVE CATEGORY — need_attention**
State the trade-off and name what to watch, without commanding.

**ALWAYS close with a soft Sylo nudge (need_attention only — REQUIRED):** Every
need_attention card MUST end its action sentence with a gentle, low-pressure
invitation to check in with the assistant **Sylo** about it. Keep it a warm,
optional-sounding offer — NEVER a command, warning, or guilt-trip. Always write
the name "Sylo" verbatim. Vary the wording month to month so it never reads
identically.

Preferred SHAPE B (2 sentences, period split) — a hedge opener softens the tone, then the action sentence names "what to watch" AND folds in the Sylo invitation:
• "It seems your work output held strong late in the month while finance flattened. Worth a quick look at your spending plan with Sylo before the next cycle."
• "It looks like your sleep hours tightened mid-month but slipped again as the meeting load climbed. Something to keep an eye on as bedtime drifts — Sylo can help you take a look."
• "It seems your savings pace trailed the monthly target through the middle weeks. Sylo can walk through it with you alongside next month's plan."
• "On busier meeting days, sleep tended to shorten. Something to watch as the schedule heats up, and Sylo's happy to dig into it with you."

(SHAPE A with comma + connective is also fine if it reads more natural in the target language.)
"""


_CATEGORY_OPPORTUNITY = """**ACTIVE CATEGORY — opportunity**
Name the shift and the chance it opens up.
YOU or a concrete behaviour is always the subject — NEVER an abstract concept (stability, imbalance, balance, control, gap, rhythm, etc.).

Preferred SHAPE B (2 sentences, period split) — a hedge opener softens the tone, then "This is a chance to ..." / "There's room to ..." transitions to the opportunity:
• "It seems you narrowed the gap between work, health, and finance from early to mid-month. This is a chance to carry that steadier mix into the closing stretch."
• "It looks like your sleep hours grew more consistent through the second half. There's room to lean on that rhythm during the next stretch of busy days."
• "It seems your spending eased back closer to plan as the month progressed. This is a chance to redirect that breathing room toward your savings goal."
• "Workouts picked back up in the closing stretch after a quieter start. There's a steadier base now to build on next month."

(SHAPE A with comma + connective is also fine if it reads more natural in the target language.)
"""


_CATEGORY_EXTENSIONS: Dict[str, str] = {
    "great_job": _CATEGORY_GREAT_JOB,
    "need_attention": _CATEGORY_NEED_ATTENTION,
    "opportunity": _CATEGORY_OPPORTUNITY,
}


def _localize_prompt(
    base_prompt: str, category: Optional[str], language: Optional[str]
) -> str:
    """Compose final system prompt: base + category extension + language directive.

    Per Path B (daily-style scoped prompts): the base carries shared rules,
    and ``_CATEGORY_EXTENSIONS[category]`` supplies the category-specific voice
    direction + concrete examples. LLM only sees examples relevant to the
    current card, which keeps the prompt focused and avoids cross-contamination
    between great_job / need_attention / opportunity voices.

    Banned-vocab list stays English (existing constraint — guards English text
    only). Template fallback also stays English; non-EN users only get
    localised text on the LLM happy-path.
    """
    language_name = get_language_name(language)
    # candidate.tentative_category is Title Case ("Need Attention") but
    # _CATEGORY_EXTENSIONS is keyed snake_case ("need_attention"); normalize so
    # the lookup hits and the category voice block is injected.
    cat_key = (category or "").strip().lower().replace(" ", "_")
    extension = _CATEGORY_EXTENSIONS.get(cat_key, "")
    parts = [base_prompt]
    if extension:
        parts.append(extension)
    parts.append(important_language_prompt(language_name))
    return "\n\n".join(parts)


# Raw-signal keys that are INTERNAL scoring/threshold values — never expose to
# the LLM. Per Rules §0 ("Threshold protection — Avoid over-triggering") and §4
# ("Language Transformation Layer"), composite scores / std-devs / correlations
# must stay internal; if surfaced, the LLM tends to render them as misleading
# percentages (e.g. balance_score=72 → "72% balance" feels meaningful even
# though the engine only treats it as a level proxy).
_INTERNAL_RAW_SIGNAL_KEYS: frozenset = frozenset(
    {
        # Module balance internals (now mean-based 0..100; still leak-prone)
        "balance_score",
        "current_balance_score",
        "prev_month_balance_score",
        "module_max_excess_pct",
        "module_scores",
        "imbalance_std",
        "phase1_balance",
        "phase2_balance",
        "avg_balance",
        "balance_drop",
        "improvement",
        "deterioration",
        "gap_vs_baseline_pct",
        "magnitude",
        "excess",
        # Phase score spread / std-dev
        "overall_day_score_std",
        "phase1_std",
        "phase2_std",
        "phase2_module_std",
        "modules_compared",
        "phase3_overall_day_score_std",
        "max_phase_sleep_std",
        "phase_score_spread_pct",
        "phase_mean_drift_pct",
        "growth",
        "variance_growth",
        # Phase-vs-phase deltas (proportions, sign confusing)
        "phase2_vs_phase1_delta",
        "phase3_vs_phase2_delta",
        "p3_vs_max_p1p2_delta",
        "p2_vs_p1_delta",
        "p3_vs_p2_delta",
        "d_sleep",
        "d_work",
        "d_act",
        "d_spend",
        "delta",
        "earlier_drop",
        "health_delta",
        "work_delta",
        "sleep_delta",
        # Pearson correlations (sign matters but raw value confusing)
        "corr_work_health",
        "corr_workout_mood",
        "corr_meeting_mood",
        "corr_work_spend",
        "corr_work_sleep",
        "corr_sleep_mood",
        "corr_workout_productivity",
        "correlation_value",
        "max_abs_corr",
        # Internal composite scores (raw 0-100 score that LLM tends to expose
        # as a meaningless number)
        "phase1_overall_avg",
        "phase1_avg",
        "phase2_avg",
        "phase1_health_avg",
        "phase2_health_avg",
        "phase1_work_avg",
        "prev_month_health_avg",
        "prev_month_work_avg",
        "prev_month_phase1_avg",
        "prev_month_phase2_avg",
        "current_month_avg",
        "prev_month_avg",
        "phase3_expense_per_day",
        "baseline_p1p2_expense_per_day",
        "p3_vs_baseline_delta",
        "phase1_overall_std",
        # Per-pattern raw measurements that the LLM tends to leak into the
        # rendered text as a "report stat" (e.g. "0.45 of days", "5.8 hours").
        # We expose qualitative labels instead (see ``_label_*`` helpers below).
        "h_sleep_hours_avg",
        "mood_range_avg",
        "max_bedtime_std_min",
        "max_wake_time_std_min",
        "p_meeting_minutes_avg",
        "heavy_meeting_days",
        "events_per_day",
        "short_events_fraction",
        "avg_event_min",
        "p_focus_blocks_per_day",
        "focus_blocks_per_day",
        "task_completion_avg",
        "task_completion_rate",
        "m_mood_score_avg",
        "max_phase_mood_std",
        "occurrences_in_last_3_months",
        "phase_score_spread",
        # Family D goal-vs-reality fractions: leak as 3 separate % per module,
        # LLM lists all three. We expose qualitative `alignment_label`,
        # `strong_areas` / `weak_areas`, and `worst_area` / `best_area` instead.
        "alignment",
        "avg_alignment",
        "aligned_modules",
        "misaligned_modules",
        # Internal classification tag
        "trigger",
        "signal_kind",
        "downgraded_from",
        "downgrade_reason",
        # Intelligence layer internals — see ``docs/INSIGHT_DETAILED_REPORT.md``.
        # ``tone_hint`` is passed to the LLM as its own directive (see
        # ``render_candidate``) so we strip it from raw_signal JSON to avoid
        # the model citing it as if it were user data. Other layer internals
        # likewise stay invisible: numeric scores would be quoted as stats,
        # classification labels would leak as jargon.
        "confidence",
        "confidence_tier",
        "memory_annotation",
        "tone_hint",
        "context.life_mode",
        "context.goal_difficulty",
        "context.growth_trend",
        "context.intentional_intensity",
        "context.reroute_from",
    }
)

# Other behaviourally-meaningful proportions stay in raw_signal — the LLM
# renders them as percentages per the prompt's "fraction → percentage" rule.
# (Goal-vs-Reality fractions previously stayed here too, but the LLM
# consistently listed all 3 module %s in one breath — Family D detectors
# now emit qualitative `alignment_label` / `strong_areas` / `weak_areas`
# / `worst_area` / `best_area` instead, and the raw map is stripped above.)
#   late_evening_day_fraction=0.3        → "on about 30% of days"


def _sanitize_raw_signal_for_llm(raw_signal: Dict[str, Any]) -> Dict[str, Any]:
    """Drop internal-only keys + rename judgmental field names before the LLM sees them (Rules §0/§4).

    The renames defuse LLM tendency to use field NAMES as descriptive labels:
    ``strong_areas: ["productivity"]`` → "productivity is strong" → translates
    harshly. Neutral names (``top_areas`` / ``lower_areas``) push the LLM to
    describe behaviour instead of label modules.
    """
    rename_map = {
        "strong_areas": "top_areas",
        "weak_areas": "lower_areas",
    }
    out: Dict[str, Any] = {}
    for k, v in raw_signal.items():
        if k in _INTERNAL_RAW_SIGNAL_KEYS:
            continue
        out[rename_map.get(k, k)] = v
    return out


# Numeric grounding validator — reject LLM output that introduces numbers not
# traceable to raw_signal. Defends against hallucinated "%" / "hours" / counts
# AND raw correlation coefficients (e.g. "r = -0.978").

# Numeric tokens to ignore (calendar artefacts, not user data):
#   - month names render as digits in some locales ("tháng 4" → "4")
#   - day-of-month "1-31" small integers in narrative phrasing
# Strategy: validate any decimal containing a dot (always a stat, never an
# index), any number paired with a unit, and skip bare small integers (1-4)
# without units.

# Capture optional leading sign so "-0.978" extracts as -0.978, not 0.978.
_NUMERIC_TOKEN_RE = re.compile(
    r"(-?\d+(?:[.,]\d+)?)\s*([kKmM]?)\s*"
    r"(%|percent|hours?|hrs?|minutes?|mins?|days?|steps?|"
    r"giờ|phút|ngày|bước|lần|sự kiện|events?)?",
    re.IGNORECASE,
)


def _collect_signal_numbers(raw_signal: Dict[str, Any]) -> List[float]:
    """Walk raw_signal recursively, collect every numeric leaf as float."""
    out: List[float] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, bool):
            return  # bools are ints in Python — skip
        elif isinstance(node, (int, float)):
            out.append(float(node))

    walk(raw_signal)
    return out


def _is_number_grounded(
    text_num: float, signal_nums: List[float], tol_rel: float = 0.10
) -> bool:
    """Return True if ``text_num`` plausibly originates from some signal_num.

    Accepts: exact match (±10% relative), sign flip, fraction→% conversion
    (0.46 ↔ 46), k-shorthand (8000 ↔ 8), m-shorthand (1_000_000 ↔ 1).

    Sign flip is allowed because the LLM may rephrase a negative correlation
    as a positive magnitude ("strong negative link, r=-0.5" → "5% drop"). The
    actual sign distinction is enforced by template phrasing, not numerics.
    """
    if not signal_nums:
        return False

    def near(a: float, b: float) -> bool:
        return abs(a - b) <= max(abs(b) * tol_rel, 0.5)

    abs_text = abs(text_num)
    for s in signal_nums:
        abs_s = abs(s)
        if near(text_num, s) or near(abs_text, abs_s):
            return True
        if near(abs_text, abs_s * 100):  # fraction → %
            return True
        if near(abs_text, abs_s / 100):  # % → fraction
            return True
        if near(abs_text, abs_s / 1000):  # 8000 → "8k"
            return True
        if near(abs_text, abs_s * 1000):  # 8 → "8000"
            return True
    return False


def _validate_numbers_grounded(text: str, raw_signal: Dict[str, Any]) -> None:
    """Reject text whose numeric tokens can't be traced to raw_signal.

    Three categories of numbers are validated:
      1. Any decimal containing a dot (e.g. ``0.978``, ``-0.5``) — these are
         always statistics, never index/date noise, so ALL must be grounded.
      2. Any number paired with a unit (``%`` / hours / minutes / steps /
         k / days / giờ / phút / ngày) — explicit user-facing stat.
      3. Any number ≥ 5 — date digits cap at 4 (months 1-12 use words in EN/VN
         monthly insights, days-of-month rare), so 5+ is almost always a stat.
    Bare small integers 1-4 without unit are skipped (calendar noise).
    """
    signal_nums = _collect_signal_numbers(raw_signal)
    for match in _NUMERIC_TOKEN_RE.finditer(text):
        raw_val, scale, unit = match.group(1), match.group(2), match.group(3)
        try:
            val = float(raw_val.replace(",", "."))
        except ValueError:
            continue
        if scale and scale.lower() == "k":
            val *= 1000.0
        elif scale and scale.lower() == "m":
            val *= 1_000_000.0
        has_unit = bool(unit) or bool(scale)
        is_decimal = "." in raw_val or "," in raw_val
        # Skip ONLY bare small integers without unit (calendar noise).
        # Decimals (statistical) and ≥5 (always stat) MUST be grounded.
        if not has_unit and not is_decimal and abs(val) < 5:
            continue
        if not _is_number_grounded(val, signal_nums):
            raise ValueError(
                f"ungrounded number {match.group(0)!r} "
                f"(extracted={val}, signal_nums={signal_nums})"
            )


async def render_candidate(
    candidate: InsightCandidate, language: Optional[str] = "en-US"
) -> str:
    """LLM-first; on failure or banned-vocab trip, fall back to template.

    Honours ``raw_signal['tone_hint']`` set by the tone layer — passed to
    the LLM as a separate directive (NOT buried inside the raw_signal JSON)
    so the model treats it as a render instruction rather than data.
    """
    safe_signal = _sanitize_raw_signal_for_llm(candidate.raw_signal)
    tone_hint = candidate.raw_signal.get("tone_hint", "moderate")
    confidence_tier = candidate.raw_signal.get("confidence_tier", "moderate")
    data_confidence = {
        "high": "High",
        "moderate": "Medium",
        "low": "Limited",
    }.get(confidence_tier, "Medium")
    # Per-pattern meaning anchor: render this pattern's (positively-framed)
    # template so the LLM sees the INTENDED idea for THIS exact pattern —
    # sharper than the category-level few-shots alone, and it keeps the LLM
    # render aligned with the template-fallback wording. Presented as MEANING
    # (not a string to copy) so it doesn't drag the output toward literal /
    # templatey phrasing, per the "translate MEANING, not words" rule in the
    # system prompt. Best-effort: a missing template / render error just omits
    # the line. Uses the full raw_signal (same as the fallback path) so the
    # reference matches what a fallback would produce.
    try:
        reference_meaning = render_template(
            candidate.raw_template_key, candidate.phase_focus, candidate.raw_signal
        )
    except Exception:
        reference_meaning = ""
    reference_line = (
        f"- reference_meaning (convey THIS idea; do NOT copy the wording, "
        f"rewrite it natively in the target language): {reference_meaning}\n"
        if reference_meaning
        else ""
    )
    user_message = (
        "Generate the insight text for this signal:\n"
        f"- pattern_family: {candidate.pattern_family}\n"
        f"- pattern_type: {candidate.pattern_type}\n"
        f"- phase_focus: {candidate.phase_focus}\n"
        f"- category: {candidate.tentative_category}\n"
        f"- tone_hint: {tone_hint}   "
        "(soft → hedge strongly; "
        "moderate → cautious factual; "
        "confident → direct; "
        "confident_sustained → emphasise continuity)\n"
        f"- data_confidence: {data_confidence}   "
        "(High → speak with confidence; "
        "Medium → standard softener only; "
        "Limited → MUST open with an explicit data-limitation marker)\n"
        f"{reference_line}"
        f"- raw_signal: {json.dumps(safe_signal, default=str)}"
    )
    system_prompt = _localize_prompt(
        _RENDER_SYSTEM_PROMPT, candidate.tentative_category, language
    )
    text = await _invoke_llm(system_prompt, user_message)
    if text:
        try:
            _validate_banned(text, source="LLM")
            _validate_numbers_grounded(text, safe_signal)
            return text
        except ValueError as e:
            logger.warning(f"⚠️ LLM output rejected, falling back to template: {e}")

    if language and not language.lower().startswith("en"):
        logger.warning(
            f"⚠️ LLM render failed for language={language!r}; "
            "returning English template fallback"
        )
    return render_template(
        candidate.raw_template_key, candidate.phase_focus, candidate.raw_signal
    )


# =============================================================================
# Service implementation
# =============================================================================


class MonthlyInsightServiceImpl(MonthlyInsightService):
    """Concrete monthly insight pipeline."""

    # TTL keeps payload alive long enough for the anti-rep lookback window
    # plus the current month being written (≈ N+1 months).
    _REDIS_TTL_SEC: int = 86400 * 31 * (ANTI_REP_LOOKBACK_MONTHS + 1)
    # Baseline snapshot TTL must cover the longest baseline (180d = 6 months)
    # plus the current month.
    _SNAPSHOT_TTL_SEC: int = 86400 * 31 * (_BASELINE_LOOKBACK_MONTHS + 1)
    # View cache for COMPLETED past months — their data + baseline are immutable,
    # so the rendered+translated response is deterministic and safe to serve
    # verbatim. Kept long (~12 months) since the content never changes.
    _VIEW_CACHE_TTL_SEC: int = 86400 * 31 * 12

    def __init__(self, external_api: IExternalAPIService) -> None:
        self.external_api: IExternalAPIService = external_api
        self.redis_client = RedisClient()
        logger.info("✅ MonthlyInsightService initialized")

    # ------------------------------------------------------------------
    # MonthlySnapshot persistence — drives BaselineContext
    # ------------------------------------------------------------------

    async def _save_monthly_snapshot(
        self, profile_id: str, snapshot: MonthlySnapshot
    ) -> None:
        """Best-effort Redis write of the raw monthly snapshot (Rules §1 cross-month)."""
        key = CacheHelpers.monthly_snapshot_redis_key(
            profile_id, snapshot.year, snapshot.month
        )
        try:
            payload = snapshot.model_dump(mode="json")
            await self.redis_client.set_data(
                key, payload, expire=self._SNAPSHOT_TTL_SEC
            )
            logger.info(
                f"💾 Saved monthly snapshot to Redis "
                f"(key={key}, ttl={self._SNAPSHOT_TTL_SEC}s)"
            )
        except Exception as e:
            logger.warning(f"⚠️ save_monthly_snapshot best-effort failed: {e}")

    async def _load_monthly_snapshot(
        self, profile_id: str, year: int, month: int
    ) -> Optional[MonthlySnapshot]:
        """Load a past MonthlySnapshot from Redis. Returns None on miss/parse error."""
        key = CacheHelpers.monthly_snapshot_redis_key(profile_id, year, month)
        try:
            payload = await self.redis_client.get_data(key)
        except Exception as e:
            logger.warning(f"⚠️ load_monthly_snapshot read failed for {key}: {e}")
            return None
        if not isinstance(payload, dict):
            return None
        try:
            return MonthlySnapshot.model_validate(payload)
        except Exception as e:
            logger.warning(f"⚠️ load_monthly_snapshot parse failed for {key}: {e}")
            return None

    async def _load_baseline_context(
        self, profile_id: str, year: int, month: int
    ) -> BaselineContext:
        """Build ``BaselineContext`` from up to 6 past months in Redis.

        Walks back ``_BASELINE_LOOKBACK_MONTHS`` and collects:
          - Prev month's MonthlySnapshot + Insight list
          - 3-month rolling average ⇒ ``avg_90d``
          - 6-month rolling average ⇒ ``avg_180d``
          - Past month → insight list mapping for pattern_repeat/reinforcement
        """
        ctx = BaselineContext()
        snapshots_in_order: List[Tuple[int, int, MonthlySnapshot]] = []
        insights_by_month: Dict[Tuple[int, int], List[Insight]] = {}

        y, m = year, month
        for step in range(1, _BASELINE_LOOKBACK_MONTHS + 1):
            m -= 1
            if m == 0:
                m = 12
                y -= 1
            snap = await self._load_monthly_snapshot(profile_id, y, m)
            if snap is not None:
                snapshots_in_order.append((y, m, snap))
                if step == 1:
                    ctx.prev_month_snapshot = snap

            # Insights for this month (drives carry-over + pattern_repeat)
            ins_key = CacheHelpers.monthly_insight_redis_key(profile_id, y, m)
            try:
                ins_payload = await self.redis_client.get_data(ins_key)
            except Exception:
                ins_payload = None
            if isinstance(ins_payload, dict):
                parsed: List[Insight] = []
                for ins_dict in ins_payload.get("insights") or []:
                    try:
                        parsed.append(Insight.model_validate(ins_dict))
                    except Exception:
                        continue
                if parsed:
                    insights_by_month[(y, m)] = parsed
                    if step == 1:
                        ctx.prev_month_insights = parsed

        # Order oldest → newest for downstream determinism
        snapshots_in_order.sort(key=lambda t: (t[0], t[1]))
        ctx.past_insights_by_month = insights_by_month

        # Compute 90d / 180d averages
        if snapshots_in_order:
            last_3 = [s for _, _, s in snapshots_in_order[-3:]]
            last_6 = [s for _, _, s in snapshots_in_order[-6:]]
            if last_3:
                ctx.avg_90d = {
                    k: _baseline_metric_avg(last_3, k) for k in _BASELINE_METRIC_KEYS
                }
            if len(last_6) >= 4:  # need ≥4 months for 180d to be meaningful
                ctx.avg_180d = {
                    k: _baseline_metric_avg(last_6, k) for k in _BASELINE_METRIC_KEYS
                }

        logger.info(
            f"📚 Baseline context for {year}-{month:02d}: "
            f"prev_month={ctx.has_prev_month()} "
            f"prev_insights={len(ctx.prev_month_insights)} "
            f"avg_90d={ctx.has_90d_baseline()} avg_180d={ctx.has_180d_baseline()} "
            f"past_insight_months={len(ctx.past_insights_by_month)}"
        )
        return ctx

    async def _save_monthly_insights(
        self,
        profile_id: str,
        year: int,
        month: int,
        insights: List[Insight],
    ) -> None:
        """Best-effort Redis write of the rendered top-N for anti-rep lookup."""
        if not insights:
            return
        key = CacheHelpers.monthly_insight_redis_key(profile_id, year, month)
        payload = {
            "year": year,
            "month": month,
            "insights": [i.model_dump(mode="json") for i in insights],
        }
        try:
            await self.redis_client.set_data(key, payload, expire=self._REDIS_TTL_SEC)
            logger.info(
                f"💾 Saved {len(insights)} monthly insights to Redis "
                f"(key={key}, ttl={self._REDIS_TTL_SEC}s)"
            )
        except Exception as e:
            logger.warning(f"⚠️ save_monthly_insights best-effort failed: {e}")

    # ------------------------------------------------------------------
    # View cache — verbatim response for COMPLETED past months
    # ------------------------------------------------------------------

    async def _load_insight_view(
        self, profile_id: str, year: int, month: int, language: str
    ) -> Optional[Dict[str, Optional[str]]]:
        """Return the cached projected response for a past month, else None."""
        key = CacheHelpers.monthly_insight_view_redis_key(
            profile_id, year, month, language
        )
        try:
            payload = await self.redis_client.get_data(key)
        except Exception as e:
            logger.warning(f"⚠️ load_insight_view read failed for {key}: {e}")
            return None
        # Only treat as a hit when the cached response actually carries at least
        # one rendered insight. An all-empty dict (a transiently-empty earlier
        # run that got cached) is treated as a MISS so it self-heals on recompute
        # instead of pinning "no insights" for the full TTL.
        if isinstance(payload, dict):
            insight = payload.get("insight")
            if isinstance(insight, dict) and any(
                isinstance(v, str) and v for v in insight.values()
            ):
                return insight
        return None

    async def _save_insight_view(
        self,
        profile_id: str,
        year: int,
        month: int,
        language: str,
        insight: Dict[str, Optional[str]],
    ) -> None:
        """Best-effort write of the final projected response for a past month."""
        key = CacheHelpers.monthly_insight_view_redis_key(
            profile_id, year, month, language
        )
        payload = {
            "year": year,
            "month": month,
            "language": language,
            "insight": insight,
        }
        try:
            await self.redis_client.set_data(
                key, payload, expire=self._VIEW_CACHE_TTL_SEC
            )
            logger.info(
                f"💾 Saved monthly insight VIEW to Redis "
                f"(key={key}, ttl={self._VIEW_CACHE_TTL_SEC}s)"
            )
        except Exception as e:
            logger.warning(f"⚠️ save_insight_view best-effort failed: {e}")

    async def _get_recent_monthly_insights(
        self,
        profile_id: str,
        before_year: int,
        before_month: int,
        lookback_months: int = ANTI_REP_LOOKBACK_MONTHS,
    ) -> List[Insight]:
        """Walk back ``lookback_months`` Redis payloads → flat list of past Insight."""
        recent: List[Insight] = []
        y, m = before_year, before_month
        for _ in range(lookback_months):
            m -= 1
            if m == 0:
                m = 12
                y -= 1
            key = CacheHelpers.monthly_insight_redis_key(profile_id, y, m)
            try:
                payload = await self.redis_client.get_data(key)
            except Exception as e:
                logger.warning(f"⚠️ get_recent_insights read failed for {key}: {e}")
                continue
            if not isinstance(payload, dict):
                continue
            for ins_dict in payload.get("insights") or []:
                try:
                    recent.append(Insight.model_validate(ins_dict))
                except Exception as e:
                    logger.warning(f"⚠️ Skipping malformed insight from {key}: {e}")
        return recent

    async def generate_monthly(
        self,
        profile_id: str,
        year: Optional[int] = None,
        month: Optional[int] = None,
        force_update: bool = False,
        language: Optional[str] = "en-US",
        timezone: Optional[str] = None,
    ) -> MonthlyInsightResponse:
        # Resolve the target month HERE (the router is a thin interface): honour
        # an explicit year+month, otherwise default to the current month in the
        # user's timezone.
        if year and month:
            if not 1 <= month <= 12:
                raise ValueError(f"month must be 1..12, got {month}")
        else:
            year, month = _current_year_month(timezone)

        # Day 1-2 (user tz): redirect a current-month target to last month so the
        # final-day synthesis has a complete month of data — explicit + derived.
        year, month = _shift_if_final_day_window(year, month, timezone)

        # True in the day-1-2 window (now in user tz is the month right after the
        # analysed one) → last month's wrap-up; restricts detection to the 5
        # final-day patterns only (see below). Day 3+ — incl. browsing an older
        # month — never trips this, so those show the Phase-3 + family view.
        final_day_only = _is_final_day_window(year, month, timezone)

        user_language = language or "en-US"
        gen_language = resolve_generation_language(user_language)

        # View cache: a COMPLETED past month (strictly before the current month,
        # not the day-1-2 final-day wrap-up) is immutable, so its rendered+
        # translated response is served verbatim — no detection / LLM / Redis
        # baseline recompute. The day-1-2 wrap-up is excluded so its (different)
        # content never poisons the past-month Phase-3 cache. ``force_update``
        # bypasses the cache to force a fresh recompute (and refresh the view).
        cur_year, cur_month = _current_year_month(timezone)
        view_cacheable = (
            (year, month) < (cur_year, cur_month)
            and not final_day_only
            and not force_update
        )
        if view_cacheable:
            cached_view = await self._load_insight_view(
                profile_id, year, month, user_language
            )
            if cached_view is not None:
                logger.info(
                    f"⚡ Monthly insight VIEW cache hit — {year}-{month:02d} "
                    f"({user_language}); skipping recompute"
                )
                return MonthlyInsightResponse(
                    status="success",
                    user_id=profile_id,
                    insight=cached_view,
                    error=None,
                )

        last_day = monthrange(year, month)[1]
        month_start = date(year, month, 1)
        month_end = date(year, month, last_day)
        logger.info(
            f"📊 Monthly insight start — profile={profile_id} "
            f"period={month_start} → {month_end} "
            f"language={language} timezone={timezone} force_update={force_update}"
        )

        # Step 1 — Fetch daily snapshots
        try:
            dailies: List[DailySnapshot] = await self.external_api.get_daily_snapshots(
                profile_id=profile_id,
                start=month_start,
                end=month_end,
                timezone=timezone,
                language=language,
            )
        except RuntimeError as e:
            logger.error(f"❌ Snapshot API failed: {e}")
            return MonthlyInsightResponse(
                status="error",
                user_id=profile_id,
                error=str(e),
            )

        # Step 2 — Phase split (always 3)
        phases = split_phases(dailies, year, month)

        # Step 3 — Monthly rollup
        snapshot: MonthlySnapshot = compute_monthly(
            dailies=dailies,
            profile_id=profile_id,
            year=year,
            month=month,
            phases=phases,
        )

        # Step 3.5 — Persist snapshot first so it's available for subsequent
        # months' baseline context (Rules §1 cross-month).
        await self._save_monthly_snapshot(profile_id, snapshot)

        # Step 3.6 — Load baseline context (prev month / 90d / 180d)
        baseline = await self._load_baseline_context(profile_id, year, month)

        # Step 4 — Pattern detection (now baseline-aware)
        candidates: List[InsightCandidate] = detect_all(snapshot, phases, baseline)
        logger.info(
            f"🔎 detect_all → {len(candidates)} candidate(s): "
            f"{[c.pattern_type for c in candidates]}"
        )

        # Day 1-2 window → ONLY the 5 final-day synthesis patterns (last month's
        # wrap-up). Every other day — including browsing an older month — drops
        # the final-day patterns and instead surfaces the active phase's patterns
        # (days 3-10 → P1, 11-20 → P2, 21-EOM → P3; a completed past month → P3)
        # plus the whole-month / cross / Family A-F patterns.
        if final_day_only:
            candidates = [
                c for c in candidates if c.pattern_type in _FINAL_DAY_PATTERN_TYPES
            ]
            logger.info(
                f"🗓️ Final-Day window — final-day patterns only: "
                f"{[c.pattern_type for c in candidates]}"
            )
        else:
            before_fd = len(candidates)
            candidates = [
                c for c in candidates if c.pattern_type not in _FINAL_DAY_PATTERN_TYPES
            ]
            if len(candidates) != before_fd:
                logger.info(
                    f"🚫 Outside day-1-2 window — suppressed "
                    f"{before_fd - len(candidates)} final-day pattern(s)"
                )
            active_phase = _active_phase_focus(year, month, timezone)
            before_pw = len(candidates)
            candidates = _apply_phase_window(candidates, active_phase)
            if len(candidates) != before_pw:
                logger.info(
                    f"🗓️ Phase window P{active_phase} — kept "
                    f"{len(candidates)}/{before_pw} candidate(s) "
                    f"(dropped other-phase patterns)"
                )

        # Step 4.1 — [Layer 2] Quality Gates: drop patterns reading
        # sparse modules or relying on suspected-corrupt mood data, and raise
        # the minimum signal floor when the month has few days of data.
        quality_flags = compute_quality_flags(snapshot, phases)
        before = len(candidates)
        candidates = apply_quality_gates(candidates, quality_flags)
        logger.info(
            f"🚧 quality_gates → {len(candidates)}/{before} "
            f"(floor={quality_flags.min_strength_floor}, "
            f"sparse={sorted(quality_flags.sparse_modules)}, "
            f"mood_stuck={quality_flags.mood_stuck})"
        )

        # Step 5 — Edge-case filter (Rules §5 — kept alongside quality gates)
        candidates = apply_filter(candidates, snapshot)

        # Step 5.1 — [Layer 3] Global gradient noise floor.
        candidates = apply_gradient_floor(candidates)

        # Step 5.2 — [Layer 5] Context Interpretation (stamps + optional re-route).
        if baseline is not None:
            ctx = compute_context(snapshot, phases, baseline)
            candidates = apply_context_interpretation(candidates, ctx)
            logger.info(
                f"🧭 context → life_mode={ctx.life_mode} "
                f"goal_difficulty={ctx.goal_difficulty} "
                f"growth_trend={ctx.growth_trend:+.2f} "
                f"intentional={ctx.intentional_intensity}"
            )

        # Step 5.3 — [Layer 9] Memory annotation (sustained/escalation/etc.).
        candidates = annotate_memory(candidates, baseline)

        # Step 5.4 — [Layer 6] Confidence scoring per candidate.
        candidates = annotate_confidence(candidates, snapshot, baseline, quality_flags)

        # Step 5.5 — [Layer 10a] Quality validator (pre-render rules).
        # May downgrade NA → Opportunity when confidence is low.
        before = len(candidates)
        candidates = filter_quality(candidates)
        if len(candidates) != before:
            logger.info(f"🪛 quality_validator → {len(candidates)}/{before}")

        # Step 5.6 — [Layer 11] Tone hint stamp (drives render tone).
        candidates = annotate_tone(candidates)

        # Step 6 — Anti-rep lookup (best-effort; failure ⇒ empty)
        recent: List[Insight] = []
        try:
            recent = await self._get_recent_monthly_insights(profile_id, year, month)
        except Exception as e:
            logger.warning(f"⚠️ Anti-rep lookup failed: {e}")

        # Step 7 — Score (signal × novelty × cross-module).
        # Recency weighting uses ``baseline.past_insights_by_month`` so a match
        # last month penalises more than one 3 months ago (Rules §4 Rule 3).
        scored: List[Tuple[InsightCandidate, float, float]] = []
        for c in candidates:
            n = novelty_score(
                c,
                recent_insights=recent,
                past_insights_by_month=baseline.past_insights_by_month,
                current_year=year,
                current_month=month,
            )
            s = compute_score(c, n)
            scored.append((c, s, n))

        # Step 7.5 — Rules §4 Rule 1: apply week-based family rotation boost.
        # No-op when the analysed month is not the current calendar month.
        scored = _apply_week_focus_boost(scored, year, month)

        # Step 7.6 — Rules §5 row 4: conflict-priority boost so
        # correlation > trend > level when candidates compete for top-N slots.
        scored = _apply_conflict_priority_boost(scored)

        # Step 8 — Dedupe
        deduped = _dedupe(scored)

        # Step 9 — Top N: prefer 1 of each category (Rules §6 final output)
        top = _pick_one_per_category(deduped, n=_TOP_N)
        top_summary = ", ".join(
            f"{c.pattern_type}→{c.tentative_category}" for c, _, _ in top
        )
        logger.info(f"🏅 Top-3 selected ({len(top)} insight(s)): [{top_summary}]")

        # Step 10 — Render text concurrently. Each candidate's LLM render is
        # independent so we ``asyncio.gather`` them: drops total render
        # latency from 3× single-call (~3-9s) to max(single-call) (~1-3s).
        # ``return_exceptions=True`` so one slow/failed call doesn't sink the
        # other two — failed slots fall back to the safe template path inline.
        async def _safe_render(c: InsightCandidate) -> str:
            try:
                return await render_candidate(c, language=gen_language)
            except Exception as e:
                logger.warning(f"⚠️ render_candidate failed for {c.pattern_type}: {e}")
                return f"({c.pattern_type})"

        rendered_texts: List[str] = await asyncio.gather(
            *(_safe_render(cand) for cand, _, _ in top)
        )

        # Step 10.5 — [Layer 10b] Post-render similarity drop (Feedback §6).
        # Compares each rendered text against the last few months' insights;
        # near-duplicates are removed so the user doesn't see the same
        # observation worded slightly differently month after month.
        if recent:
            recent_texts = [ins.text for ins in recent if getattr(ins, "text", None)]
            rendered_pairs = [(t[0], text) for t, text in zip(top, rendered_texts)]
            kept_pairs = filter_rendered_similarity(rendered_pairs, recent_texts)
            if len(kept_pairs) != len(rendered_pairs):
                kept_keys = {c.pattern_type for c, _ in kept_pairs}
                top = [t for t in top if t[0].pattern_type in kept_keys]
                rendered_texts = [text for _cand, text in kept_pairs]
                logger.info(
                    f"🪛 similarity_filter → kept {len(top)}/{len(rendered_pairs)}"
                )

        insights: List[Insight] = []
        for idx, ((cand, score, novelty), text) in enumerate(zip(top, rendered_texts)):
            insights.append(
                Insight(
                    rank=idx + 1,
                    category=cand.tentative_category,
                    pattern_family=cand.pattern_family,
                    pattern_type=cand.pattern_type,
                    phase_focus=cand.phase_focus,
                    text=text,
                    score=score,
                    signal_strength=cand.signal_strength,
                    novelty=novelty,
                    cross_module_impact=cand.cross_module_impact,
                    raw_signal=cand.raw_signal,
                    insight_hash=insight_hash(cand),
                )
            )

        # Step 11 — Best-effort persist (full Insight rows saved for anti-rep)
        await self._save_monthly_insights(profile_id, year, month, insights)

        # Step 12 — Project to {great_job, need_attention, opportunity} response shape
        projected = _project_insights(insights)
        projected_for_translate = {
            k: v for k, v in projected.items() if isinstance(v, str) and v
        }
        if projected_for_translate:
            translated = await translate_insight_dict(
                projected_for_translate,
                user_language,
                _LLMHolder.get(),
                insight_type="monthly_insight",
            )
            projected = {**projected, **translated}

        # Cache the final response for a completed past month so repeat calls
        # short-circuit at the top. Skipped for the current (still-changing)
        # month, the day-1-2 final-day wrap-up, and EMPTY results — caching an
        # empty response would pin "no insights" for the whole TTL even if it was
        # only a transient miss (sparse data, baseline not yet warm). force_update
        # recompute still refreshes the stored view here.
        if insights and (year, month) < (cur_year, cur_month) and not final_day_only:
            await self._save_insight_view(
                profile_id, year, month, user_language, projected
            )

        return MonthlyInsightResponse(
            status="success",
            user_id=profile_id,
            insight=projected,
            error=None,
        )


def _project_insights(insights: List[Insight]) -> Dict[str, Optional[str]]:
    """Project ranked Insight list to {great_job, need_attention, opportunity}.

    Each value is the rendered text of the highest-ranked insight in that
    category, or ``None`` if the category produced no candidate.
    """
    out: Dict[str, Optional[str]] = {
        "great_job": None,
        "need_attention": None,
        "opportunity": None,
    }
    by_category = {
        "Great Job": "great_job",
        "Need Attention": "need_attention",
        "Opportunity": "opportunity",
    }
    for ins in insights:
        key = by_category.get(ins.category)
        if key and out[key] is None:
            out[key] = ins.text
    return out

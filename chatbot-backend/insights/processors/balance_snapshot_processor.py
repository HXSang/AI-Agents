import logging
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Union

from insights.schemas.processed_context import BalanceSnapshotBlock

logger = logging.getLogger(__name__)
GOOD_DAY_THRESHOLD = 70.0


class BalanceSnapshotProcessor:
    @staticmethod
    def process(raw_data: Dict[str, Any]) -> Optional[BalanceSnapshotBlock]:
        today = _extract_today_score(raw_data)
        week = _extract_week_scores(raw_data.get("balance_scores_30d"))
        today_has_data = any(today.get(k) is not None for k in (
            "productivity", "health", "finance", "balance",
        ))
        if not today_has_data and not week.get("balance"):
            return None

        week_avg = _mean(week.get("balance"))
        prod_week_avg = _mean(week.get("productivity"))
        health_week_avg = _mean(week.get("health"))
        finance_week_avg = _mean(week.get("finance"))
        week_std = _stdev(week.get("balance"))
        good_days = sum(
            1
            for v in week.get("balance")
            if v is not None and v >= GOOD_DAY_THRESHOLD
        )
        trend_delta = _compute_trend_delta(week.get("balance"))

        return BalanceSnapshotBlock(
            productivity_score=today.get("productivity"),
            health_score=today.get("health"),
            finance_score=today.get("finance"),
            balance_score=today.get("balance"),
            balance_date=today.get("date"),
            balance_timezone=today.get("timezone"),
            balance_score_7d_avg=week_avg,
            productivity_score_7d_avg=prod_week_avg,
            health_score_7d_avg=health_week_avg,
            finance_score_7d_avg=finance_week_avg,
            balance_score_7d_std=week_std,
            balance_score_trend_delta=trend_delta,
            balance_score_7d_good_days=good_days,
            balance_score_7d_total_days=len(week.get("balance") or []),
        )


# ──────────────────────────────────────────────────────────────────────
# Helpers — small, defensive, and pure so tests can call them directly.
# ──────────────────────────────────────────────────────────────────────


def _coerce_score_entry(
    item: Any,
) -> Optional[Dict[str, Any]]:
    if isinstance(item, dict):
        return item
    if hasattr(item, "model_dump"):
        try:
            return item.model_dump()
        except Exception:
            return None
    return None


def _safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _extract_today_score(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    payload = _coerce_score_entry(raw_data.get("balance_score"))
    if not payload:
        return {}

    return {
        "productivity": _safe_float(
            payload.get("productivityScore") or payload.get("productivity_score")
        ),
        "health": _safe_float(
            payload.get("healthScore") or payload.get("balance_health_score") or payload.get("health_score")
        ),
        "finance": _safe_float(
            payload.get("financeScore") or payload.get("finance_score")
        ),
        "balance": _safe_float(
            payload.get("balanceScore") or payload.get("balance_score")
        ),
        "date": payload.get("balance_score_date") or payload.get("date"),
        "timezone": payload.get("timezone"),
    }


def _extract_week_scores(
    raw_range: Optional[List[Any]],
) -> Dict[str, List[Optional[float]]]:
    if not isinstance(raw_range, list) or not raw_range:
        return {"balance": [], "productivity": [], "health": [], "finance": []}

    series: Dict[str, List[Optional[float]]] = {
        "balance": [],
        "productivity": [],
        "health": [],
        "finance": [],
    }
    for raw in raw_range:
        row = _coerce_score_entry(raw)
        if not row:
            continue
        series["balance"].append(
            _safe_float(row.get("balanceScore") or row.get("balance_score"))
        )
        series["productivity"].append(
            _safe_float(
                row.get("productivityScore") or row.get("productivity_score")
            )
        )
        series["health"].append(
            _safe_float(row.get("healthScore") or row.get("balance_health_score") or row.get("health_score"))
        )
        series["finance"].append(
            _safe_float(row.get("financeScore") or row.get("finance_score"))
        )
    return series


def _mean(values: List[Optional[float]]) -> Optional[float]:
    """Mean of numeric values, ignoring None. None for empty/all-None."""
    nums = [v for v in values if isinstance(v, (int, float))]
    if not nums:
        return None
    return round(sum(nums) / len(nums), 1)


def _stdev(values: List[Optional[float]]) -> Optional[float]:
    """Population std-dev. None for empty or 1-element series."""
    nums = [v for v in values if isinstance(v, (int, float))]
    if len(nums) < 2:
        return None
    mean = sum(nums) / len(nums)
    var = sum((v - mean) ** 2 for v in nums) / len(nums)
    return round(var ** 0.5, 2)


def _compute_trend_delta(values: List[Optional[float]]) -> Optional[float]:
    nums = [v for v in values if isinstance(v, (int, float))]
    if len(nums) < 4:
        return None
    mid = len(nums) // 2
    first = sum(nums[:mid]) / mid
    second = sum(nums[mid:]) / (len(nums) - mid)
    return round(second - first, 1)
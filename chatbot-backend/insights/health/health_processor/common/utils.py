"""Shared helpers for health signal processors."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional


def is_data_stale(staleness_days: Optional[int], threshold: int = 0) -> bool:
    """Return True if data is stale (not from today)."""
    if staleness_days is None:
        return True
    return staleness_days > threshold


def safe_get(obj: Any, *keys, default=None):
    cur = obj
    for k in keys:
        if cur is None:
            return default
        try:
            if isinstance(cur, dict):
                cur = cur.get(k, default)
            else:
                cur = getattr(cur, k, default)
        except Exception:
            return default
        if cur is None:
            return default
    return cur


def snapshot_field(item: Any, field: str) -> Any:
    """Read a field off a snapshot row regardless of dict or object shape."""
    if isinstance(item, dict):
        return item.get(field)
    return getattr(item, field, None)


def snapshot_date(item: Any) -> Optional[date]:
    """Parse daily_snapshot_date (post-canonicalize key) to a date object."""
    raw = snapshot_field(item, "daily_snapshot_date")
    if not raw:
        return None
    try:
        if isinstance(raw, date) and not isinstance(raw, datetime):
            return raw
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).date()
    except Exception:
        return None


def health_param_value(
    health_params: Optional[Any],
    canonical_key: str,
    legacy_key: str,
) -> Any:
    if health_params is None:
        return None
    if isinstance(health_params, dict):
        v = health_params.get(canonical_key)
        if v is None:
            v = health_params.get(legacy_key)
        return v
    v = getattr(health_params, canonical_key, None)
    if v is None:
        v = getattr(health_params, legacy_key, None)
    return v

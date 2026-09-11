"""Strip empty / null values from API payloads before they reach the insight pipeline.

Why this exists
---------------
Upstream endpoints occasionally return ``{}``, ``[]``, ``null``, or whitespace-only
strings for fields that have no real data. The naive pipeline kept those values in
``raw_data`` and forwarded the whole blob to the LLM, where:

1. They wasted tokens on placeholder structure (``{"steps_today": null}`` etc.)
2. They confused the model — an empty field still primed the LLM to "say
   something" about it, which is a classic hallucination path.
3. They made staleness / freshness heuristics noisier: an empty ``signals`` block
   cannot be distinguished from "the device synced but reported nothing".

This helper runs **at the API collector boundary** so downstream processors and
LLM prompts see a clean payload. It is opt-in per payload (default = on) and
always logs what it dropped so we can audit regressions.

Design choices
--------------
- **Drop empty containers at any depth** (``{}``, ``[]``, ``null``, ``""``,
  whitespace strings). 0 is kept (a real ``steps_today=0`` is meaningful, not a
  placeholder).
- **Keep empty lists at the top level when the rest of the pipeline relies on
  the key being present.** A few processors (e.g. ``today_reminders``) treat
  ``[]`` as "API responded successfully with no items" vs missing key. We expose
  ``keep_top_level_list_keys`` for those cases.
- **Optionally preserve empty items inside lists** (e.g. a calendar event that
  becomes ``{}`` after cleaning still gets kept as ``{}`` so the row survives
  and downstream code can see "this event exists"). Toggle with
  ``preserve_empty_items=True`` and restrict to specific parent keys via
  ``keep_empty_item_keys`` (default = every list inherits the behavior).
- **Do NOT mutate the caller's dict** — return a fresh dict so caller-side
  retries / debug dumps remain intact.
- **Never raise.** A bad payload should still flow through and degrade the
  insight; it should not break collection.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

DEFAULT_KEEP_TOP_LEVEL_LIST_KEYS: frozenset[str] = frozenset(
    {
        "today_reminders",
        "calendar_events",
        "calendar_events_week",
        "moods_7d",
        "productivity_summaries_7d",
        "balance_scores_30d",
        "finance_goals",
        "finance_bills",
        "finance_logs",
        "finance_budgets",
        "goal_contributions",
    }
)

DISPLAY_DROP_EMPTY_LIST_KEYS: frozenset[str] = frozenset({"historical_snapshots"})

def _is_empty_scalar(value: Any) -> bool:
    """Return True for values that carry no information worth keeping."""
    if value is None:
        return True
    if isinstance(value, str):
        # Whitespace-only strings are equivalent to missing.
        return value.strip() == ""
    return False


def _should_preserve_empty_item(
    parent_key: Optional[str],
    *,
    preserve_empty_items: bool,
    keep_empty_item_keys: Optional[Iterable[str]],
) -> bool:
    """Decide whether a list item that cleaned to ``{}`` should be kept."""
    if not preserve_empty_items:
        return False
    kept = set(keep_empty_item_keys or ())
    if not kept:
        return True
    return parent_key is not None and parent_key in kept


def clean_value(
    value: Any,
    *,
    keep_top_level_list_keys: Optional[Iterable[str]] = None,
    is_top_level: bool = False,
    preserve_empty_items: bool = False,
    keep_empty_item_keys: Optional[Iterable[str]] = None,
    in_list_item: bool = False,
    parent_key: Optional[str] = None,
) -> Tuple[Any, int]:
    kept_list_keys: Set[str] = set(keep_top_level_list_keys or ())

    if _is_empty_scalar(value):
        return None, 1

    if isinstance(value, dict):
        cleaned: Dict[str, Any] = {}
        dropped = 0
        for k, v in value.items():
            child, child_dropped = clean_value(
                v,
                keep_top_level_list_keys=keep_top_level_list_keys,
                is_top_level=False,
                preserve_empty_items=preserve_empty_items,
                keep_empty_item_keys=keep_empty_item_keys,
                parent_key=k,
            )
            if child is None and not (is_top_level and isinstance(v, list) and k in kept_list_keys):
                dropped += 1 + child_dropped
                continue
            cleaned[k] = child
            dropped += child_dropped

        if not cleaned:
            # An empty dict inside a list item: the item itself is now {}
            if in_list_item and _should_preserve_empty_item(
                parent_key,
                preserve_empty_items=preserve_empty_items,
                keep_empty_item_keys=keep_empty_item_keys,
            ):
                return {}, dropped
            return None, dropped
        return cleaned, dropped

    if isinstance(value, list):
        cleaned_list: List[Any] = []
        dropped = 0
        for item in value:
            child, child_dropped = clean_value(
                item,
                keep_top_level_list_keys=keep_top_level_list_keys,
                is_top_level=False,
                preserve_empty_items=preserve_empty_items,
                keep_empty_item_keys=keep_empty_item_keys,
                in_list_item=True,
                parent_key=parent_key,  # the parent dict key that owns this list
            )
            if child is None:
                dropped += 1 + child_dropped
                continue
            cleaned_list.append(child)
            dropped += child_dropped
        if not cleaned_list:
            # Top-level: caller decides whether to keep `[]`.
            return None, dropped
        return cleaned_list, dropped

    # Numbers (incl. 0), bools, datetimes, model instances, etc. — keep as-is.
    return value, 0


def clean_payload(
    payload: Any,
    *,
    keep_top_level_list_keys: Optional[Iterable[str]] = None,
    preserve_empty_items: bool = False,
    keep_empty_item_keys: Optional[Iterable[str]] = None,
    log_tag: str = "empty_cleanser",
    log_drop_threshold: int = 1,
) -> Any:
    """Strip empty / null values from an API payload dict.

    Parameters
    ----------
    payload:
        Dict returned by an upstream API call (or any JSON-compatible tree).
    keep_top_level_list_keys:
        Iterable of keys whose empty-list value should survive at the top
        level. See ``DEFAULT_KEEP_TOP_LEVEL_LIST_KEYS`` for the defaults used
        by the data collector.
    preserve_empty_items:
        When True, list items that become empty after cleaning are kept
        (returned as ``{}``). Use this when a row needs to survive even if
        every field is null/empty — e.g. a calendar event you still want to
        surface.
    keep_empty_item_keys:
        Restrict ``preserve_empty_items`` to a specific set of parent list
        keys. ``None`` means every list inherits the behavior. Pass an
        iterable to scope it (e.g. ``["calendar_events"]``).
    log_tag:
        Prefix for the log line. Useful when several collectors run per
        pipeline call.
    log_drop_threshold:
        Only emit a log line when at least this many values were dropped.
        Set to ``0`` to always log; set to a large number to silence.

    Returns
    -------
    Cleaned copy of ``payload``. The original is not mutated. If ``payload``
    is not a dict, it is returned as-is (and logged).
    """
    # Default behaviour: keep a curated set of top-level empty lists (e.g.
    # today_reminders=[], mood_7d=[]) so downstream processors can distinguish
    # "API returned empty" from "key absent".  Pass keep_top_level_list_keys
    # to override.  NOTE: an empty frozenset() is *not* falsy in this context
    # only when it is the explicit intent — we distinguish None (use default)
    # from frozenset() (keep nothing).
    if keep_top_level_list_keys is not None:
        kept = set(keep_top_level_list_keys)
    else:
        kept = set(DEFAULT_KEEP_TOP_LEVEL_LIST_KEYS)

    if not isinstance(payload, dict):
        if _is_empty_scalar(payload):
            logger.info(
                "[%s] payload is empty (%s); returning as-is",
                log_tag,
                type(payload).__name__,
            )
        return payload

    cleaned: Dict[str, Any] = {}
    total_dropped = 0
    top_level_empties: List[str] = []

    for k, v in payload.items():
        child, child_dropped = clean_value(
            v,
            keep_top_level_list_keys=kept,
            is_top_level=True,
            preserve_empty_items=preserve_empty_items,
            keep_empty_item_keys=keep_empty_item_keys,
            parent_key=k,
        )
        if child is None:
            # Decide: top-level empty list on a whitelisted key survives as `[]`.
            if isinstance(v, list) and k in kept:
                cleaned[k] = []
                top_level_empties.append(f"{k}[]")
                continue
            total_dropped += 1 + child_dropped
            top_level_empties.append(k)
            continue
        cleaned[k] = child
        total_dropped += child_dropped

    if total_dropped >= log_drop_threshold and top_level_empties:
        sample = top_level_empties[:8]
        suffix = f" (+{len(top_level_empties) - 8} more)" if len(top_level_empties) > 8 else ""
        logger.info(
            "[%s] dropped %d empty value(s); sample keys=%s%s",
            log_tag,
            total_dropped,
            sample,
            suffix,
        )

    return cleaned


def drop_empty_lists(
    payload: Any,
    *,
    keys: Iterable[str],
) -> Any:
    """Drop top-level keys whose value is an empty list.

    Unlike passing those keys through ``clean_payload(keep_top_level_list_keys=...)``,
    which removes the key **regardless of contents**, this helper preserves
    non-empty lists so debug panels can still show real data (e.g.
    ``historical_snapshots`` with 30 days of entries).

    Parameters
    ----------
    payload:
        Dict to scrub. Non-dicts are returned unchanged.
    keys:
        Top-level keys to check. Only an ``== []`` value triggers a drop; a
        dict value is never touched.

    Returns
    -------
    A new dict with the named empty-list keys removed. Non-list values
    (including populated dicts, scalars, and missing keys) are preserved.
    """
    if not isinstance(payload, dict):
        return payload
    target = set(keys)
    if not target:
        return dict(payload)
    out: Dict[str, Any] = {}
    for k, v in payload.items():
        if k in target and isinstance(v, list) and len(v) == 0:
            continue
        out[k] = v
    return out
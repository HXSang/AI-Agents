"""
Deterministic mapper: smart_suggestions (plan mode) → multi-action queue.

Turns the structured ``action_params`` of each plan-mode suggestion into the
``{action_id, tool_name, arguments}`` shape consumed by the multi-action
pipeline (``execute_single_action``), WITHOUT another LLM round-trip.

Source schema  : app/ai_agents/tools/executor/smart_suggestions/schemas.py
                 (ActionParams.type ∈ {create_event, update_event, delete_event,
                  set_health_goal, create_finance_log})
Target tools   : create_event_by_name   (calendar_tools.py)
                 update_event_by_name    (calendar_tools.py) — move/reschedule
                 delete_event_by_name    (calendar_tools.py) — remove
                 set_health_goal         (health_tools.py)
                 create_finance_logs     (finance_tool.py)
"""

from typing import Any, Dict, List, Optional

from app.ai_agents.utils.time_utils import to_local_wall_clock
from app.utils.logger import logger


def _to_naive_local(value: Any, timezone: Optional[str]) -> Any:
    """Drop the tz offset, KEEPING the wall-clock → naive local 'YYYY-MM-DDTHH:MM'
    (thin wrapper over `time_utils.to_local_wall_clock`, which carries the full
    keep-the-clock rationale). `timezone` is unused, kept for signature parity;
    non-string / unparseable input is returned unchanged.
    """
    if not isinstance(value, str) or not value:
        return value
    return to_local_wall_clock(value) or value


# action_params.type → target tool name
_TYPE_TO_TOOL = {
    "create_event": "create_event_by_name",
    "set_health_goal": "set_health_goal",
    "create_finance_log": "create_finance_logs",
    "update_event": "update_event_by_name",
    "delete_event": "delete_event_by_name",
    "create_reminder": "create_reminder",
}


def _reminder_dict(ap: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """create_reminder → one reminder dict {title, due_date, has_time, category,
    notes} (NO provider_name — that is the action-level arg). Requires title + due_date;
    returns None if either is missing. Used for both single and batched (reminders=[])
    forms, mirroring _calendar_event_dict."""
    title = ap.get("summary")
    due = ap.get("due_date")
    if not title or not due:
        return None
    r: Dict[str, Any] = {"title": title, "due_date": due}
    if ap.get("has_time") is not None:
        r["has_time"] = ap["has_time"]
    if ap.get("category"):
        r["category"] = ap["category"]
    extra = ap.get("extra") or {}
    if extra.get("notes"):
        r["notes"] = extra["notes"]
    return r


def _reminder_args(ap: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """create_reminder → create_reminder(provider_name, title, due_date, has_time, …)
    single-reminder arg form."""
    r = _reminder_dict(ap)
    if not r:
        return None
    return {"provider_name": "insidesync", **r}


def _calendar_event_dict(ap: Dict[str, Any]) -> Dict[str, Any]:
    """create_event → one event dict {summary, start_time, end_time, category,
    description, location} (NO provider_name — that is the action-level arg).

    Used both for the single-event arg form and for batching multiple creates
    into one create_event_by_name(events=[...]) call.
    """
    extra = ap.get("extra") or {}
    ev: Dict[str, Any] = {}
    if ap.get("summary"):
        ev["summary"] = ap["summary"]
    if ap.get("start_time"):
        ev["start_time"] = ap["start_time"]
    if ap.get("end_time"):
        ev["end_time"] = ap["end_time"]
    if ap.get("category"):
        ev["category"] = ap["category"]
    if extra.get("description"):
        ev["description"] = extra["description"]
    if extra.get("location"):
        ev["location"] = extra["location"]
    return ev


def _calendar_args(ap: Dict[str, Any]) -> Dict[str, Any]:
    """create_event → create_event_by_name(provider_name, summary, start_time,
    end_time, category, description, location) — single-event arg form."""
    return {"provider_name": "insidesync", **_calendar_event_dict(ap)}


def _health_goal_args(ap: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """set_health_goal → set_health_goal(metric, target_value, unit).

    Both metric and target_value are required by the tool; skip if missing.
    """
    metric = ap.get("metric")
    target_value = ap.get("target_value")
    if not metric or target_value is None:
        return None
    extra = ap.get("extra") or {}
    args: Dict[str, Any] = {"metric": metric, "target_value": target_value}
    if extra.get("unit"):
        args["unit"] = extra["unit"]
    return args


def _finance_args(ap: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """create_finance_log → create_finance_logs(entries=[{amount, description,
    date, currency}]).

    Amount is required for a meaningful log; skip if absent.
    """
    extra = ap.get("extra") or {}
    amount = extra.get("amount")
    if amount is None:
        amount = ap.get("target_value")
    if amount is None:
        return None
    entry: Dict[str, Any] = {"amount": amount}
    if ap.get("summary"):
        entry["description"] = ap["summary"]
    if ap.get("start_time"):
        entry["date"] = ap["start_time"]
    if extra.get("currency"):
        entry["currency"] = extra["currency"]
    return {"entries": [entry]}


def _update_event_args(ap: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """update_event → update_event_by_name(event_name, start_time, end_time,
    start_date, end_date).

    `event_name` (existing event's summary) is required to locate the event.
    `start_time`/`end_time` are the NEW times. `start_date`/`end_date` are
    derived from the event's ORIGINAL day to narrow the LLM name-matching to the
    correct day (reduces wrong-event matches). Skip if event_name missing.
    """
    event_name = ap.get("event_name")
    if not event_name:
        return None
    args: Dict[str, Any] = {
        "provider_name": "insidesync",
        "event_name": event_name,
    }
    if ap.get("start_time"):
        args["start_time"] = ap["start_time"]
    if ap.get("end_time"):
        args["end_time"] = ap["end_time"]
    if ap.get("summary"):
        args["summary"] = ap["summary"]
    if ap.get("category"):
        args["category"] = ap["category"]
    # Narrow event search to the original event's day.
    anchor = ap.get("original_start_time") or ap.get("start_time")
    if anchor and len(anchor) >= 10:
        args["start_date"] = anchor[:10]
        args["end_date"] = anchor[:10]
    return args


def _delete_event_args(ap: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """delete_event → delete_event_by_name(event_name=[name], start_date, end_date).

    `event_name` (existing event's summary) is required to locate the event; the
    tool takes a LIST of names. `start_date`/`end_date` come from the event's
    ORIGINAL day so only THAT occurrence is removed (not every same-named event).
    Skip if event_name missing.
    """
    event_name = ap.get("event_name")
    if not event_name:
        return None
    args: Dict[str, Any] = {
        "provider_name": "insidesync",
        "event_name": [event_name],
    }
    anchor = ap.get("original_start_time") or ap.get("start_time")
    if anchor and len(anchor) >= 10:
        args["start_date"] = anchor[:10]
        args["end_date"] = anchor[:10]
    return args


_ARG_BUILDERS = {
    "create_event": _calendar_args,
    "set_health_goal": _health_goal_args,
    "create_finance_log": _finance_args,
    "update_event": _update_event_args,
    "delete_event": _delete_event_args,
    "create_reminder": _reminder_args,
}


def normalize_action_queue_times(
    queue: List[Dict[str, Any]], timezone: Optional[str]
) -> List[Dict[str, Any]]:
    """Strip tz offsets → naive local on every time field in `queue` (in place).

    The calendar API expects a naive local time + a separate `timezone` field; an
    offset-aware string makes it land one offset off (see _to_naive_local). Applied
    to BOTH the initial build AND free-text edits so an edited event never
    reintroduces the offset. Covers flat create, batched events[], and update_event
    start/end. Date-only fields (start_date/end_date) are left untouched.
    """
    for action in queue or []:
        if not isinstance(action, dict):
            continue
        args = action.get("arguments") or {}
        for key in ("start_time", "end_time", "due_date"):
            if key in args:
                args[key] = _to_naive_local(args[key], timezone)
        for ev in args.get("events") or []:
            if isinstance(ev, dict):
                for key in ("start_time", "end_time"):
                    if key in ev:
                        ev[key] = _to_naive_local(ev[key], timezone)
        for r in args.get("reminders") or []:
            if isinstance(r, dict) and "due_date" in r:
                r["due_date"] = _to_naive_local(r["due_date"], timezone)
    return queue


def build_action_queue_from_suggestions(
    suggestions: List[Dict[str, Any]],
    timezone: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Build a multi-action queue from plan-mode suggestions.

    Args:
        suggestions: ``suggestions[]`` from the smart_suggestions JSON envelope
            (each item has ``action_params``).
        timezone: kept for signature parity with the rest of the pipeline;
            suggestion timestamps are already ISO-8601 with offset.

    Returns:
        List of ``{action_id, tool_name, arguments}`` — ready for
        ``execute_single_action``. Suggestions without a valid/executable
        ``action_params`` are skipped (logged), so the queue may be shorter
        than the input or empty.
    """
    # E1: all create_event suggestions are merged into ONE
    # create_event_by_name(events=[...]) action so the user confirms ONE step
    # instead of N (multi-action pauses between each action). Non-create types
    # (set_health_goal, create_finance_log, update_event, delete_event) stay as
    # separate actions — update/delete in particular handle one event per call.
    create_events: List[Dict[str, Any]] = []
    reminders: List[Dict[str, Any]] = []
    delete_actions: List[Dict[str, Any]] = []
    update_actions: List[Dict[str, Any]] = []
    other_actions: List[Dict[str, Any]] = []  # set_health_goal, create_finance_log

    for suggestion in suggestions or []:
        ap = (suggestion or {}).get("action_params") or {}
        ap_type = ap.get("type")

        tool_name = _TYPE_TO_TOOL.get(ap_type)
        builder = _ARG_BUILDERS.get(ap_type)
        if not tool_name or builder is None:
            logger.warning(
                f"[SS-HANDOFF] step 4 | skip suggestion: unknown "
                f"action type {ap_type!r}"
            )
            continue

        if ap_type == "create_event":
            ev = _calendar_event_dict(ap)
            if not ev.get("start_time"):
                logger.warning(
                    f"[SS-HANDOFF] step 4 | skip create_event: missing "
                    f"start_time (ap={ap})"
                )
                continue
            create_events.append(ev)
            continue

        if ap_type == "create_reminder":
            r = _reminder_dict(ap)
            if not r:
                logger.warning(
                    f"[SS-HANDOFF] step 4 | skip create_reminder: missing "
                    f"title/due_date (ap={ap})"
                )
                continue
            reminders.append(r)
            continue

        arguments = builder(ap)
        if not arguments:
            logger.warning(
                f"[SS-HANDOFF] step 4 | skip suggestion: {ap_type!r} "
                f"missing required fields (ap={ap})"
            )
            continue
        entry = {"tool_name": tool_name, "arguments": arguments}
        if ap_type == "delete_event":
            delete_actions.append(entry)
        elif ap_type == "update_event":
            update_actions.append(entry)
        else:
            other_actions.append(entry)

    queue: List[Dict[str, Any]] = []

    # Reshape the calendar FIRST, then add: delete (free a slot) and update (move an
    # event to its new time) run BEFORE the new create_events, so the additions land
    # on the already-cleared/reshaped day — matching how the plan was reasoned.
    queue.extend(delete_actions)
    queue.extend(update_actions)

    # Batched create: single event keeps the flat arg form (unchanged single-create
    # behaviour); 2+ events go through the `events` batch param.
    if len(create_events) == 1:
        queue.append(
            {
                "tool_name": "create_event_by_name",
                "arguments": {"provider_name": "insidesync", **create_events[0]},
            }
        )
    elif len(create_events) > 1:
        queue.append(
            {
                "tool_name": "create_event_by_name",
                "arguments": {
                    "provider_name": "insidesync",
                    "events": create_events,
                },
            }
        )

    # Batched reminders: single keeps the flat arg form; 2+ go through `reminders`.
    if len(reminders) == 1:
        queue.append(
            {
                "tool_name": "create_reminder",
                "arguments": {"provider_name": "insidesync", **reminders[0]},
            }
        )
    elif len(reminders) > 1:
        queue.append(
            {
                "tool_name": "create_reminder",
                "arguments": {"provider_name": "insidesync", "reminders": reminders},
            }
        )

    # Independent goals/logs last (no calendar-time dependency).
    queue.extend(other_actions)

    normalize_action_queue_times(queue, timezone)

    # Assign sequential action_ids over the final queue.
    for idx, action in enumerate(queue):
        action["action_id"] = f"action_{idx + 1}"

    logger.info(
        f"[SS-HANDOFF] step 4 | built {len(queue)} action(s) from "
        f"{len(suggestions or [])} suggestion(s) "
        f"(batched {len(create_events)} create_event): "
        f"tool_names={[a['tool_name'] for a in queue]}"
    )
    return queue

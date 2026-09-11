"""Suggestion-plan handoff.

Extracted from hybrid_workflow to keep the orchestrator lean. One entry point
used by the workflow's thin node wrapper:

- ``build_suggestion_handoff`` — turn a smart_suggestions(plan) result into a
  multi-action DRAFT + advisory message (the SS-handoff).

It takes the workflow's LLM/memory and a few of its helper callables so the
behaviour is identical to the in-class method it replaced.
"""

from typing import Any, Callable, Dict, List, Optional

from app.ai_agents.constants import QuickReply
from app.ai_agents.multi_action.multi_action_extractor import plan_multi_actions
from app.ai_agents.multi_action.suggestion_to_actions import (
    build_action_queue_from_suggestions,
)
from app.ai_agents.prompt import format_current_time_prompt
from app.utils.logger import logger


async def build_suggestion_handoff(
    state: Dict[str, Any],
    *,
    llm: Any,
    memory_service: Any,
    extract_text_fn: Callable[[Any], str],
    fallback_fn: Callable[[str], str],
    create_buttons_fn: Callable[[List[Dict[str, str]], str], List[Dict[str, str]]],
) -> Optional[Dict[str, Any]]:
    """Hand a smart_suggestions(plan) result into the multi-action draft/confirm flow.

    Returns the response dict (draft saved, advisory shown, Continue/Cancel
    buttons) when the agent produced plan-mode suggestions with executable
    actions; returns ``None`` to signal the caller to fall through to the normal
    LLM multi-action extract.
    """
    ss = state.get("smart_suggestions_data")
    if not (ss and ss.get("mode") == "plan"):
        return None

    user_id = state.get("user_id", "")
    session_id = state.get("session_id", "")
    timezone = state.get("timezone")
    language = state.get("language_name", "English")
    language_code = state.get("language_code", "en")
    user_message = state.get("user_message", "")

    action_queue = build_action_queue_from_suggestions(
        ss.get("suggestions") or [], timezone
    )
    if not action_queue:
        # No executable action_params → fall through to normal LLM extract.
        logger.info(
            "[SS-HANDOFF] step 4 | no executable action_params -> "
            "fallback to LLM multi-action extract"
        )
        return None

    # Persists the DRAFT queue (side effect) so the user's Continue can execute
    # it later, and returns the rendered plan text.
    plan_text = await plan_multi_actions(
        user_id=user_id,
        session_id=session_id,
        user_message=user_message,
        action_queue=action_queue,
        llm=llm,
        memory_service=memory_service,
        extract_text_fn=extract_text_fn,
        language_name=language,
        current_time=format_current_time_prompt(timezone),
    )
    # Prefer the agent's own advisory (Option A). But if the model jumped
    # straight to the tool with no advisory text, execute_agent set
    # agent_response to the generic greeting fallback — that would mask the plan,
    # so use plan_text. A non-empty response is also required:
    # _route_after_extract_multi treats empty as "auto-execute now".
    agent_advisory = (state.get("agent_response") or "").strip()
    fallback_greeting = fallback_fn(language).strip()
    if not agent_advisory or agent_advisory == fallback_greeting:
        response_text = plan_text
    else:
        response_text = agent_advisory
    buttons = create_buttons_fn(
        [
            {
                "value": QuickReply.MULTI_ACTION_CONTINUE,
                "label_key": QuickReply.LABEL_KEY_CONFIRM,
            },
            {
                "value": QuickReply.MULTI_ACTION_CANCEL,
                "label_key": QuickReply.LABEL_KEY_CANCEL,
            },
        ],
        language_code,
    )
    logger.info(
        f"[SS-HANDOFF] step 5 | advisory shown ({len(action_queue)} "
        f"action(s) drafted) -> await user Continue/Cancel to execute"
    )
    return {
        "multi_action_mode": True,
        "action_queue": action_queue,
        "complete_action_queue": [],
        "response": response_text,
        "quick_reply_buttons": buttons,
    }

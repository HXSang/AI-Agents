"""
Multi-action executor: pops and executes each action from the action_queue.

HybridWorkflow imports execute_single_action and calls it inside
_execute_single_action_node — no execution logic lives there.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage

from app.ai_agents.constants import (
    ICONS,
    QUICK_REPLY_BUTTON_LABELS,
    BotConfig,
    QuickReply,
    SmartSuggestionsConstants,
    ToolStatus,
)
from app.ai_agents.middleware.ai_middleware import (
    AIOutputValidationMiddleware,
    AIToolValidationMiddleware,
    RealValueContextMiddleware,
    ReasoningContentFilterMiddleware,
    SuggestionHandoffMiddleware,
    ToolBoundaryMiddleware,
    ToolChoicesValidationMiddleware,
)
from app.ai_agents.middleware.retry_middleware import RetryModelCallMiddleware
from app.ai_agents.multi_action.multi_action_prompt import (
    extract_operations_prompt,
    multi_action_cancel_prompt,
    multi_action_connective_tissue_prompt,
    multi_action_empty_queue_prompt,
    multi_action_plan_prompt,
    multi_action_single_action_system_prompt,
    multi_action_skip_connective_prompt,
    multi_action_summary_prompt,
)
from app.ai_agents.multi_action.multi_action_store import (
    delete_action_queues,
    get_action_queues,
    get_and_clear_draft_queue,
    save_action_queues,
    save_draft_queue,
)
from app.ai_agents.multi_action.suggestion_to_actions import (
    build_action_queue_from_suggestions,
)
from app.ai_agents.pending_action import PendingActionService
from app.ai_agents.prompt import format_current_time_prompt, workflow_system_prompt
from app.ai_agents.schemas.workflow_schemas import RealValueContext
from app.ai_agents.tools import (
    create_balance_tools,
    create_calendar_tools,
    create_company_info_tools,
    create_finance_tools,
    create_health_tools,
    create_productivity_tools,
    create_profile_tools,
    create_reminder_tools,
    create_search_tools,
    create_smart_suggestions_tools,
)
from app.ai_agents.tools.validate.pending_action_manager import PendingActionManager
from app.services.memory_service import IMemoryService
from app.utils.logger import logger

# ─────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────


async def handle_multi_action_quick_reply(
    state: Dict[str, Any],
    quick_reply: Dict[str, str],
    llm: Any,
    memory_service: IMemoryService,
    extract_text_fn: Callable[[Any], str],
    language_names: Dict[str, str],
) -> Dict[str, Any]:
    """Orchestrator: Coordinates quick reply handling for multi-action."""
    quick_reply_value = quick_reply.get("value", "")

    if quick_reply_value == QuickReply.MULTI_ACTION_CANCEL:
        return await process_multi_action_cancel(
            state, quick_reply, llm, memory_service, extract_text_fn, language_names
        )

    if quick_reply_value == QuickReply.MULTI_ACTION_CONTINUE:
        return await process_multi_action_continue(
            state, quick_reply, memory_service, language_names
        )

    return {}


async def process_multi_action_cancel(
    state: Dict[str, Any],
    quick_reply: Dict[str, str],
    llm: Any,
    memory_service: IMemoryService,
    extract_text_fn: Callable[[Any], str],
    language_names: Dict[str, str],
) -> Dict[str, Any]:
    user_id = state.get("user_id", "")
    session_id = state.get("session_id", "")
    language_code = quick_reply.get("language_code") or state.get("language_code", "en")
    language_name = language_names.get(language_code, "English")

    draft_data = await get_and_clear_draft_queue(user_id, session_id, memory_service)

    is_initial_cancel = False
    if draft_data:
        draft_queue = draft_data.get("action_queue", [])
        if draft_queue:
            is_initial_cancel = True
            logger.info(
                f"[multi_action] CANCEL — {len(draft_queue)} initial draft actions aborted"
            )

            cancel_response = await build_cancel_response(
                llm=llm,
                extract_text_fn=extract_text_fn,
                complete_action_queue=[],
                remaining_queue=draft_queue,
                language_name=language_name,
            )

            return {
                "language_code": language_code,
                "language_name": language_name,
                "multi_action_mode": False,
                "action_queue": [],
                "complete_action_queue": [],
                "quick_reply_result": {"status": "multi_action_cancelled"},
                "response": cancel_response,
            }

    # SCENARIO 2: Skip current action during execution flow
    if not is_initial_cancel:
        existing_queues = await get_action_queues(user_id, session_id, memory_service)
        if existing_queues:
            action_queue = existing_queues.get("action_queue", [])
            complete_action_queue = existing_queues.get("complete_action_queue", [])
            original_message = existing_queues.get("original_message", "")

            if action_queue:
                _pam = PendingActionManager()
                current_tool = action_queue[0].get("tool_name")
                if current_tool:
                    _pam.delete(user_id, session_id, current_tool)

                skipped_action = action_queue.pop(0)
                skipped_action["result"] = "Skipped by user."
                skipped_action["status"] = "cancelled"
                complete_action_queue.append(skipped_action)

                logger.info(
                    f"[multi_action] SKIP — Action '{skipped_action.get('tool_name')}' skipped. {len(action_queue)} actions remaining."
                )

                await save_action_queues(
                    user_id=user_id,
                    session_id=session_id,
                    action_queue=action_queue,
                    complete_action_queue=complete_action_queue,
                    memory_service=memory_service,
                    original_message=original_message,
                )

                return {
                    "language_code": language_code,
                    "language_name": language_name,
                    "multi_action_mode": True,
                    "action_queue": action_queue,
                    "complete_action_queue": complete_action_queue,
                }
            else:
                await delete_action_queues(user_id, session_id, memory_service)

    # Fallback response
    return {
        "language_code": language_code,
        "language_name": language_name,
        "multi_action_mode": False,
        "action_queue": [],
        "complete_action_queue": [],
        "quick_reply_result": {"status": "multi_action_cancelled"},
        "response": "Action cancelled.",
    }


async def process_multi_action_continue(
    state: Dict[str, Any],
    quick_reply: Dict[str, str],
    memory_service: IMemoryService,
    language_names: Dict[str, str],
) -> Dict[str, Any]:

    user_id = state.get("user_id", "")
    session_id = state.get("session_id", "")
    language_code = quick_reply.get("language_code") or state.get("language_code", "en")
    language_name = language_names.get(language_code, "English")

    draft_data = await get_and_clear_draft_queue(user_id, session_id, memory_service)

    if draft_data:
        action_queue = draft_data.get("action_queue", [])
        complete_action_queue = []
        original_message = draft_data.get("original_message", "")
        logger.info(
            f"[multi_action] CONTINUE (Step 1) — Moving {len(action_queue)} actions from Draft to Pending."
        )
    else:
        existing_queues = await get_action_queues(user_id, session_id, memory_service)
        if existing_queues:
            action_queue = existing_queues.get("action_queue", [])
            complete_action_queue = existing_queues.get("complete_action_queue", [])
            original_message = existing_queues.get("original_message", "")
            logger.info(
                f"[multi_action] CONTINUE (Next Steps) — Resuming {len(action_queue)} pending actions."
            )
        else:
            logger.warning("[multi_action] CONTINUE — Queue not found or expired")
            return {
                "multi_action_mode": False,
                "action_queue": [],
                "complete_action_queue": [],
            }

    # A delete/update front action only caches need_confirmation; its write runs via
    # PendingActionService, which the CONTINUE path never calls. Execute the cached
    # pending here so it isn't re-cached forever, then advance the queue.
    if user_id and action_queue:
        front_tool = (action_queue[0] or {}).get("tool_name", "")
        if front_tool in ("delete_event_by_name", "update_event_by_name"):
            cached = PendingActionManager().get(user_id, session_id, front_tool)
            if cached and cached.get("status") == ToolStatus.NEED_CONFIRMATION:
                logger.info(
                    f"[multi_action] CONTINUE — executing gated front action "
                    f"'{front_tool}' (cached need_confirmation)"
                )
                exec_result = await PendingActionService(
                    user_id,
                    session_id,
                    state.get("user_data", {}) or {},
                    timezone=state.get("timezone"),
                ).handle_quick_reply(
                    {
                        "value": QuickReply.CONFIRM_REQUEST,
                        "language_code": language_code,
                    }
                )
                exec_status = (exec_result or {}).get("status", ToolStatus.SUCCESS)
                logger.info(
                    f"[multi_action] gated front action '{front_tool}' "
                    f"executed -> {exec_status}"
                )
                done = action_queue.pop(0)
                done["result"] = "Executed via multi-action confirm."
                done["status"] = exec_status
                complete_action_queue.append(done)

    if user_id and (action_queue or complete_action_queue):
        await save_action_queues(
            user_id=user_id,
            session_id=session_id,
            action_queue=action_queue,
            complete_action_queue=complete_action_queue,
            memory_service=memory_service,
            original_message=original_message,
        )

    return {
        "language_code": language_code,
        "language_name": language_name,
        "multi_action_mode": True,
        "action_queue": action_queue,
        "complete_action_queue": complete_action_queue,
    }


async def execute_single_action(
    state: Dict[str, Any],
    llm: Any,
    validator_llm: Any,
    memory_service: IMemoryService,
    extract_text_fn: Callable[[Any], str],
) -> Dict[str, Any]:
    action_queue: List[Dict[str, Any]] = list(state.get("action_queue") or [])
    complete_action_queue: List[Dict[str, Any]] = list(
        state.get("complete_action_queue") or []
    )

    if not action_queue:
        logger.warning("[execute_single_action] action_queue is empty")
        language = state.get("language_name", "English")
        return {"response": await _fallback(llm, extract_text_fn, language)}

    current_action = action_queue[0]
    tool_name: str = current_action.get("tool_name", "none")
    arguments: Dict[str, Any] = current_action.get("arguments") or {}
    action_id: str = current_action.get("action_id", "action_1")

    user_id: str = state.get("user_id", "")
    session_id: str = state.get("session_id", "")
    user_data: Dict[str, Any] = state.get("user_data", {})
    timezone: Optional[str] = state.get("timezone")
    language: str = state.get("language_name", "English")
    language_code: str = state.get("language_code", "en")
    bot_name: str = state.get("bot_name", BotConfig.DEFAULT_BOT_NAME)

    logger.info(f"[execute_single_action] [{action_id}] tool={tool_name}")

    # ── GENERATE SKIP PREFIX IF PREVIOUS ACTION WAS JUST CANCELLED ──
    quick_reply = state.get("quick_reply", {})
    qr_value = quick_reply.get("value", "")
    skip_prefix = ""

    if qr_value == QuickReply.MULTI_ACTION_CANCEL and complete_action_queue:
        last_completed = complete_action_queue[-1]
        if last_completed.get("status") == "cancelled":
            skipped_tool = last_completed.get("tool_name", "")

            skipped_arguments = last_completed.get("arguments") or {}

            logger.info(
                f"[execute_single_action] Generating skip transition from '{skipped_tool}' to '{tool_name}' and PAUSING execution."
            )

            skip_text = await _generate_skip_connective_tissue(
                llm,
                extract_text_fn,
                skipped_tool,
                skipped_arguments,
                tool_name,
                arguments,
                language,
            )

            if not skip_text:
                skip_text = await _skip_fallback(llm, extract_text_fn, language)

            labels = QUICK_REPLY_BUTTON_LABELS.get(
                language_code, QUICK_REPLY_BUTTON_LABELS.get("en", {})
            )
            return {
                "action_queue": action_queue,
                "complete_action_queue": complete_action_queue,
                "response": skip_text,
                "quick_reply_buttons": [
                    {
                        "value": QuickReply.MULTI_ACTION_CONTINUE,
                        "label": labels.get(QuickReply.LABEL_KEY_CONFIRM, "Continue"),
                        "language_code": language_code,
                    },
                    {
                        "value": QuickReply.MULTI_ACTION_CANCEL,
                        "label": labels.get(QuickReply.LABEL_KEY_CANCEL, "Cancel"),
                        "language_code": language_code,
                    },
                ],
                "has_pending_confirmation": True,
            }

    all_tools = _build_tools(
        user_id, session_id, user_data, timezone, language, state, llm
    )

    if tool_name.lower() == "none":
        tools = []
        fell_back = False
        logger.info(
            f"[execute_single_action] tool is 'none'. Bypassing fallback and executing pure LLM chat."
        )
    else:
        tools = [t for t in all_tools if t.name == tool_name]
        fell_back = False
        if not tools:
            logger.warning(
                f"[execute_single_action] tool '{tool_name}' not found — falling back"
            )
            tools = all_tools
            fell_back = True

    current_time_str = format_current_time_prompt(timezone)
    user_message_context = state.get("user_message", "")

    description = current_action.get("description", "No description provided.")

    if fell_back:
        system_prompt = workflow_system_prompt(
            current_time=current_time_str,
            language=language,
            user_profile=str(user_data),
            bot_name=bot_name,
        )
        user_msg = f"User's original request: '{user_message_context}'. Please handle the appropriate action that matches your tools."
    else:
        system_prompt = multi_action_single_action_system_prompt(
            tool_name=tool_name,
            arguments=arguments,
            description=description,
            language=language,
            current_time=current_time_str,
            bot_name=bot_name,
        )
        user_msg = (
            f"Task: Execute tool '{tool_name}'.\n"
            f"Initial arguments from planner: {arguments}\n"
            f"Original request: '{user_message_context}'\n\n"
            f"CRITICAL INSTRUCTION: Before calling the tool, you MUST review the chat history to find missing parameters (like time, date, location). Combine them with the initial arguments!"
        )

    # ── EXECUTE AGENT ───────────────────────────────────────────────────
    chat_history: List[Dict[str, Any]] = state.get("chat_history", [])
    is_suggestion = tool_name == SmartSuggestionsConstants.TOOL_NAME
    result_text, exec_status, sugg_envelope = await _run_agent(
        llm=llm,
        validator_llm=validator_llm,
        tools=tools,
        system_prompt=system_prompt,
        user_msg=user_msg,
        chat_history=chat_history,
        extract_text_fn=extract_text_fn,
        language=language,
        action_id=action_id,
        needs_real_ids=True,
        timezone=timezone,
        bot_name=bot_name,
        capture_suggestion=is_suggestion,
    )

    has_pending_confirmation = False
    auto_confirm_sync_available = False
    auto_confirm_sync_event_count = 0

    if user_id:
        _pam = PendingActionManager()
        tools_to_check = [t.name for t in tools]

        for t_name in tools_to_check:
            _cached = _pam.get(user_id, session_id, t_name)
            if _cached:
                _pstatus = _cached.get("status")
                valid_pause_statuses = [
                    "need_confirmation",
                    "suggest",
                    "missing",
                    "editing",
                    "overlapping",
                    "confirm_cancel",
                    "ask_send_email",
                ]
                if _pstatus in valid_pause_statuses:
                    has_pending_confirmation = True
                    tool_name = t_name
                    logger.info(
                        f"[execute_single_action] [{action_id}] Pausing — ACTUAL tool '{tool_name}' has status='{_pstatus}', waiting for user interaction"
                    )
                    break

    # ── DECIDE THE FATE OF THE ACTION ──────────────────────────────────
    if not has_pending_confirmation:
        logger.info(
            f"[execute_single_action] [{action_id}] FULLY COMPLETED. Popping from queue."
        )
        completed_action = action_queue.pop(0)
        completed_action["result"] = result_text
        completed_action["status"] = exec_status
        complete_action_queue.append(completed_action)

        # Nested smart_suggestions (plan mode): expand the running queue with
        # its suggested actions so they execute through the same per-action
        # Continue gate + write validators. improve/evaluate carry no
        # action_params -> no splice (text answer stands).
        if sugg_envelope and sugg_envelope.get("mode") == "plan":
            nested = build_action_queue_from_suggestions(
                sugg_envelope.get("suggestions") or [], timezone
            )
            if nested:
                action_queue = nested + action_queue
                logger.info(
                    f"[execute_single_action] [{action_id}] nested suggestion "
                    f"-> spliced {len(nested)} action(s) to front; queue now "
                    f"{len(action_queue)}"
                )
    else:
        logger.info(
            f"[execute_single_action] [{action_id}] SUSPENDED. Keeping in queue."
        )
        action_queue[0]["tool_name"] = tool_name
        action_queue[0]["result"] = result_text
        action_queue[0]["status"] = exec_status

    # ── PERSIST BOTH QUEUES TO REDIS ──────────────────────────────────────
    if user_id:
        await save_action_queues(
            user_id=user_id,
            session_id=session_id,
            action_queue=action_queue,
            complete_action_queue=complete_action_queue,
            memory_service=memory_service,
            original_message=user_message_context,
        )

    # ── BUILD RESPONSE + TRANSITION ────────────────────────────────────
    final_response = skip_prefix + result_text

    if action_queue and not has_pending_confirmation:
        logger.info(
            f"[execute_single_action] {len(action_queue)} action(s) remaining — waiting"
        )
        next_action = action_queue[0]
        connective_sentence = await _generate_connective_tissue(
            llm, extract_text_fn, result_text, next_action, language
        )

        if connective_sentence:
            final_response = f"{connective_sentence}"
        else:
            next_tool_name = next_action.get("tool_name", "")
            fallback_sentence = await _proceed_fallback(
                llm, extract_text_fn, next_tool_name, language
            )
            final_response = f"{fallback_sentence}"

    elif has_pending_confirmation:
        logger.info(
            f"[execute_single_action] [{action_id}] need_confirmation pending — pausing multi-action flow"
        )
    else:
        logger.info("[execute_single_action] queue empty — proceeding to summary")

    return {
        "action_queue": action_queue,
        "complete_action_queue": complete_action_queue,
        "response": final_response,
        "quick_reply_buttons": None,
        "has_pending_confirmation": has_pending_confirmation,
        "auto_confirm_sync_available": auto_confirm_sync_available,
        "auto_confirm_sync_event_count": auto_confirm_sync_event_count,
    }


def extract_multi_actions(
    user_message: str,
    intent: Dict[str, Any],
    llm_invoke: Callable,
    extract_text_fn: Callable[[Any], str],
    parse_json_fn: Callable[[str], Dict[str, Any]],
    available_tools_description: str = "",
    current_time: str = "",
    resolved_dates_section: str = "",
    chat_history_str: str = "",
) -> Dict[str, Any]:
    prompt = extract_operations_prompt(
        user_query=user_message,
        available_tools_description=available_tools_description,
        classification_result=intent,
        current_time=current_time,
        resolved_dates_section=resolved_dates_section,
        chat_history_str=chat_history_str,
    )
    try:
        response = llm_invoke([HumanMessage(content=prompt)])
        data = parse_json_fn(extract_text_fn(response.content))
        operations = data.get("operations", [])
        if not operations:
            return {
                "multi_action_mode": False,
                "action_queue": [],
                "complete_action_queue": [],
            }
        return {
            "multi_action_mode": True,
            "action_queue": operations,
            "complete_action_queue": [],
        }
    except Exception as e:
        logger.warning(f"[extract_multi_actions] failed: {e}")
        return {
            "multi_action_mode": False,
            "action_queue": [],
            "complete_action_queue": [],
        }


async def plan_multi_actions(
    user_id: str,
    session_id: str,
    user_message: str,
    action_queue: List[Dict[str, Any]],
    llm: Any,
    memory_service: Any,
    extract_text_fn: Callable[[Any], str],
    language_name: str,
    current_time: str = "",
) -> str:
    if user_id:
        await save_draft_queue(
            user_id=user_id,
            session_id=session_id,
            action_queue=action_queue,
            memory_service=memory_service,
            original_message=user_message,
        )

    prompt = multi_action_plan_prompt(
        user_message, action_queue, language_name, current_time
    )
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return extract_text_fn(response.content).strip()


async def build_cancel_response(
    llm: Any,
    extract_text_fn: Callable[[Any], str],
    complete_action_queue: List[Dict[str, Any]],
    remaining_queue: List[Dict[str, Any]],
    language_name: str,
) -> str:
    """
    Generate a cancellation summary message via LLM in the user's language.

    Args:
        llm:                   LLM instance.
        extract_text_fn:       Text extractor from LLM response content.
        complete_action_queue: Actions that were already executed before cancel.
        remaining_queue:       Actions that were skipped due to cancellation.
        language_name:         Full language name (e.g. 'Vietnamese', 'English', 'Japanese').

    Returns:
        A natural language cancellation summary string.
    """
    done_count = len(complete_action_queue)
    skip_count = len(remaining_queue)

    completed_summary = (
        "\n".join(
            f"- {a.get('tool_name', 'unknown')}: {a.get('result', '')[:120]}"
            for a in complete_action_queue
        )
        if complete_action_queue
        else "None"
    )
    skipped_summary = (
        "\n".join(f"- {a.get('tool_name', 'unknown')}" for a in remaining_queue)
        if remaining_queue
        else "None"
    )

    prompt = multi_action_cancel_prompt(
        done_count=done_count,
        completed_summary=completed_summary,
        skip_count=skip_count,
        skipped_summary=skipped_summary,
        language_name=language_name,
    )
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        text = extract_text_fn(response.content).strip()
        logger.info(f"[build_cancel_response] Generated ({len(text)} chars)")
        return text
    except Exception as e:
        logger.warning(f"[build_cancel_response] LLM failed, using fallback: {e}")
        if done_count == 0:
            return (
                f"Workflow cancelled. No actions were performed. ({skip_count} skipped)"
            )
        return f"Workflow cancelled. {done_count} action(s) completed, {skip_count} skipped."


async def build_empty_queue_response(
    llm: Any,
    extract_text_fn: Callable[[Any], str],
    language_name: str,
) -> str:
    """
    Generate a 'queue already empty' message via LLM in the user's language.

    This is an edge case shown when CONTINUE is received but Redis has no
    remaining actions (e.g. race condition or stale button press).

    Args:
        llm:             LLM instance.
        extract_text_fn: Text extractor from LLM response content.
        language_name:   Full language name (e.g. 'Vietnamese', 'English', 'Japanese').

    Returns:
        A natural language confirmation string.
    """
    prompt = multi_action_empty_queue_prompt(language_name)
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        text = extract_text_fn(response.content).strip()
        logger.info(f"[build_empty_queue_response] Generated ({len(text)} chars)")
        return text
    except Exception as e:
        logger.warning(f"[build_empty_queue_response] LLM failed, using fallback: {e}")
        return "All actions have already been completed."


async def summarize_multi_actions(
    state: Dict[str, Any],
    llm: Any,
    memory_service: IMemoryService,
    extract_text_fn: Callable[[Any], str],
) -> Dict[str, Any]:
    complete_action_queue: List[Dict[str, Any]] = list(
        state.get("complete_action_queue") or []
    )
    language_name: str = state.get("language_name", "English")
    language_code: str = state.get("language_code", "en")
    user_id: str = state.get("user_id", "")
    session_id: str = state.get("session_id", "")

    prompt = multi_action_summary_prompt(complete_action_queue, language_name)
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        summary = extract_text_fn(response.content).strip()
    except Exception as e:
        logger.warning(f"[summarize_multi_actions] LLM failed: {e}")
        summary = ""

    if not summary:
        summary = await _fallback(llm, extract_text_fn, language_name)

    if user_id:
        await delete_action_queues(user_id, session_id, memory_service)
        logger.info(f"[summarize_multi_actions] Cleaned up Redis for user {user_id}")
    quick_reply_buttons = None
    sync_available = state.get("auto_confirm_sync_available", False)
    sync_event_count = state.get("auto_confirm_sync_event_count", 0)

    if not sync_available and user_id:
        try:
            _pam = PendingActionManager()
            sync_data = _pam.get_sync_pending(user_id, session_id)
            if sync_data:
                sync_available = True
                sync_event_count = len(sync_data.get("event_ids", []))
        except Exception as _e:
            logger.warning(
                f"[summarize_multi_actions] Could not check sync_pending: {_e}"
            )

    if sync_available and sync_event_count > 0:
        labels = QUICK_REPLY_BUTTON_LABELS.get(
            language_code, QUICK_REPLY_BUTTON_LABELS["en"]
        )
        quick_reply_buttons = [
            {
                "value": "sync_to_google",
                "label": labels.get("sync_to_google", "Google Calendar"),
                "language_code": language_code,
                **(
                    {"icon": ICONS["sync_to_google"]}
                    if "sync_to_google" in ICONS
                    else {}
                ),
            },
            {
                "value": "no_sync",
                "label": labels.get("no_sync", "Skip"),
                "language_code": language_code,
                **({"icon": ICONS["no_sync"]} if "no_sync" in ICONS else {}),
            },
        ]
        logger.info(
            f"[summarize_multi_actions] sync_pending found — appending sync buttons"
        )

    return {
        "response": summary,
        "multi_action_mode": False,
        "quick_reply_buttons": quick_reply_buttons,
    }


# ─────────────────────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────────────────────


def _build_tools(
    user_id: str,
    session_id: str,
    user_data: Dict[str, Any],
    timezone: Optional[str],
    language: str,
    state: Dict[str, Any],
    llm: Any,
    suggestions_llm: Any = None,
) -> List[Any]:
    """
    Build the full tool set — mirrors _execute_agent_node in HybridWorkflow.

    Returns an empty list when user_id is not provided.
    """
    if not user_id:
        return []

    user_message: str = state.get("user_message", "")
    chat_history: List[Dict[str, Any]] = state.get("chat_history", [])

    ss_llm = suggestions_llm or llm

    tools: List[Any] = []
    tools.extend(create_health_tools(user_id, session_id, timezone=timezone))
    tools.extend(
        create_calendar_tools(
            user_id,
            session_id,
            user_data=user_data,
            timezone=timezone,
            user_message=user_message,
            llm=llm,
        )
    )
    tools.extend(
        create_reminder_tools(
            user_id,
            session_id,
            user_data=user_data,
            timezone=timezone,
            user_message=user_message,
            llm=llm,
        )
    )
    tools.extend(create_company_info_tools(user_id, session_id, user_message))
    tools.extend(
        create_profile_tools(
            user_id, session_id, user_data=user_data, timezone=timezone
        )
    )
    tools.extend(
        create_productivity_tools(
            user_id, session_id, user_data=user_data, timezone=timezone
        )
    )
    tools.extend(
        create_balance_tools(
            user_id, session_id, user_data=user_data, timezone=timezone
        )
    )
    tools.extend(
        create_finance_tools(
            user_id, session_id, user_data=user_data, timezone=timezone, llm=llm
        )
    )
    tools.extend(
        create_smart_suggestions_tools(
            user_id,
            session_id,
            user_data=user_data,
            timezone=timezone,
            language_name=language,
            user_message=user_message,
            bot_name=state.get("bot_name", BotConfig.DEFAULT_BOT_NAME),
            llm=ss_llm,
        )
    )
    tools.extend(
        create_search_tools(
            user_id=user_id,
            timezone=timezone,
            language=language,
            user_data=user_data,
            llm=llm,
        )
    )
    return tools


def _build_agent_message(tool_name: str, arguments: Dict[str, Any]) -> str:
    """
    Build an instruction message so the agent knows which tool to call and with what parameters.

    Filters out None values to keep the message clean.
    """
    clean_args = {k: v for k, v in (arguments or {}).items() if v is not None}
    if clean_args:
        parts = ", ".join(f"{k}: {v}" for k, v in clean_args.items())
        return f"Execute {tool_name} — {parts}"
    return f"Execute {tool_name}"


async def _run_agent(
    llm: Any,
    validator_llm: Any,
    tools: List[Any],
    system_prompt: str,
    user_msg: str,
    extract_text_fn: Callable[[Any], str],
    language: str,
    action_id: str,
    chat_history: Optional[List[Dict[str, Any]]] = None,
    needs_real_ids: bool = False,
    timezone: Optional[str] = None,
    bot_name: str = "Sylo",
    capture_suggestion: bool = False,
) -> tuple[str, str, Optional[Dict[str, Any]]]:
    """
    Run the agent and return (result_text, status, suggestion_envelope).

    suggestion_envelope is the parsed smart_suggestions JSON when
    capture_suggestion=True and the agent called the tool; otherwise None.
    """
    try:
        real_value_context = RealValueContext()
        middleware = [
            RealValueContextMiddleware(real_value_context),
            AIToolValidationMiddleware(validator_llm, real_value_context),
            AIOutputValidationMiddleware(
                validator_llm,
                real_value_context,
                language,
                needs_real_ids,
                timezone,
                bot_name,
            ),
            ReasoningContentFilterMiddleware(),
            ToolChoicesValidationMiddleware(tools=tools),
            ToolBoundaryMiddleware(),
            RetryModelCallMiddleware(),
        ]
        # Capture-only: lets the caller expand the multi-action queue from a
        # plan-mode suggestion. Safe here — the single-action agent holds one
        # non-write tool, so the middleware's block/sibling guards are no-ops.
        if capture_suggestion:
            middleware.append(SuggestionHandoffMiddleware(real_value_context))
        agent = create_agent(
            llm,
            tools,
            system_prompt=system_prompt,
            middleware=middleware,
        )
        messages = [
            (
                HumanMessage(content=msg["content"])
                if msg["role"] == "user"
                else AIMessage(content=msg["content"])
            )
            for msg in (chat_history or [])
        ]
        messages.append(HumanMessage(content=user_msg))

        result = await agent.ainvoke({"messages": messages})
        result_messages = result.get("messages", [])

        text = ""
        if result_messages:
            last = result_messages[-1]
            content = last.content if hasattr(last, "content") else str(last)
            text = extract_text_fn(content)

        if not text:
            text = await _fallback(llm, extract_text_fn, language)

        logger.info(f"[_run_agent] [{action_id}] OK — {len(text)} chars")
        return text, "success", real_value_context.smart_suggestions_data

    except Exception as e:
        logger.exception(f"[_run_agent] [{action_id}] failed: {e}")
        return await _error(llm, extract_text_fn, language), "error", None


async def _generate_connective_tissue(
    llm: Any,
    extract_text_fn: Callable[[Any], str],
    current_result: str,
    next_action: Dict[str, Any],
    language: str,
) -> str:
    """Generate a transition sentence between two consecutive actions."""
    prompt = multi_action_connective_tissue_prompt(
        current_result=current_result,
        next_tool_name=next_action.get("tool_name", ""),
        next_arguments=next_action.get("arguments") or {},
        next_description=next_action.get("description", "No description provided."),
        language=language,
    )

    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        tissue = extract_text_fn(response.content).strip().strip('"')
        logger.info(f"[_generate_connective_tissue] '{tissue}'")
        return tissue
    except Exception as e:
        logger.warning(f"[_generate_connective_tissue] failed: {e}")
        return ""


async def _proceed_fallback(
    llm: Any,
    extract_text_fn: Callable[[Any], str],
    next_tool_name: str,
    language: str,
) -> str:
    """
    Generate a 'proceed to next step' transition sentence via LLM in the user's language.

    Used as fallback when _generate_connective_tissue fails (LLM exception).
    """
    try:
        response = await llm.ainvoke(
            [
                HumanMessage(
                    content=f"Write one short friendly sentence in {language} asking if the user wants to proceed with the next step: '{next_tool_name}'. Plain text only, no markdown."
                )
            ]
        )
        return extract_text_fn(response.content).strip()
    except Exception:
        return f"Proceed with: {next_tool_name}?"


async def _fallback(
    llm: Any, extract_text_fn: Callable[[Any], str], language: str
) -> str:
    """Generate a fallback message when the action_queue is unexpectedly empty."""
    try:
        response = await llm.ainvoke(
            [
                HumanMessage(
                    content=f"Write one short friendly sentence in {language} asking how you can help the user. Plain text only."
                )
            ]
        )
        return extract_text_fn(response.content).strip()
    except Exception:
        return "How can I help you?"


async def _error(llm: Any, extract_text_fn: Callable[[Any], str], language: str) -> str:
    """Generate an error message via LLM in the user's language."""
    try:
        response = await llm.ainvoke(
            [
                HumanMessage(
                    content=f"Write one short friendly apology sentence in {language} saying something went wrong and asking the user to try again. Plain text only."
                )
            ]
        )
        return extract_text_fn(response.content).strip()
    except Exception:
        return "Sorry, something went wrong. Please try again."


async def _generate_skip_connective_tissue(
    llm: Any,
    extract_text_fn: Callable[[Any], str],
    skipped_tool: str,
    skipped_arguments: Dict[str, Any],
    next_tool: str,
    next_arguments: Dict[str, Any],
    language: str,
) -> str:
    """Generate a transition sentence when an action is skipped mid-flow."""
    try:
        prompt = multi_action_skip_connective_prompt(
            skipped_tool, skipped_arguments, next_tool, next_arguments, language
        )
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        tissue = extract_text_fn(response.content).strip().strip('"')

        logger.info(f"[_generate_skip_connective_tissue] '{tissue[:80]}'")
        return tissue

    except Exception as e:
        logger.warning(f"[_generate_skip_connective_tissue] failed: {e}")
        return ""


async def _skip_fallback(
    llm: Any, extract_text_fn: Callable[[Any], str], language: str
) -> str:
    """Generate a fallback skip message via LLM, with an absolute static fallback."""
    try:
        response = await llm.ainvoke(
            [
                HumanMessage(
                    content=f"Write one short friendly sentence in {language} confirming that the previous action was skipped and asking if we should proceed with the next task. Plain text only."
                )
            ]
        )
        return extract_text_fn(response.content).strip()
    except Exception as e:
        logger.warning(f"[_skip_fallback] LLM failed again: {e}")
        return "Action skipped. Shall we proceed with the next task?"

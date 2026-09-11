import asyncio
import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict

import pycountry
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph

from app.ai_agents.constants import (
    ICONS,
    LANGUAGE_NAMES,
    MemoryConstants,
    QUICK_REPLY_BUTTON_LABELS,
    BotConfig,
    CommonConstant,
    HybridStateConstants,
    PendingActionTool,
    QuickReply,
    ToolStatus,
    get_goal_key_from_metric,
    get_token_quota_message,
)
from app.ai_agents.llm_manager import LLMManager
from app.ai_agents.memory import get_memory_runtime_service
from app.ai_agents.memory.prompt_context import build_merged_user_context
from app.ai_agents.memory.service import LTMWriteResult
from app.ai_agents.middleware.ai_middleware import (
    AIOutputValidationMiddleware,
    AIToolValidationMiddleware,
    RealValueContextMiddleware,
    ReasoningContentFilterMiddleware,
    SuggestionHandoffMiddleware,
    ToolBoundaryMiddleware,
    ToolChoicesValidationMiddleware,
    ToolProgressMiddleware,
)
from app.ai_agents.middleware.retry_middleware import RetryModelCallMiddleware
from app.ai_agents.multi_action.draft_edit import build_suggestion_handoff
from app.ai_agents.multi_action.multi_action_extractor import (
    _build_tools,
    _generate_connective_tissue,
    _proceed_fallback,
    build_cancel_response,
    execute_single_action,
    extract_multi_actions,
    handle_multi_action_quick_reply,
    plan_multi_actions,
    process_multi_action_cancel,
    process_multi_action_continue,
    summarize_multi_actions,
)
from app.ai_agents.multi_action.multi_action_store import (
    delete_action_queues,
    get_action_queues,
    get_and_clear_draft_queue,
    save_action_queues,
)
from app.ai_agents.pending_action import PendingActionService
from app.ai_agents.progress_labels import PROGRESS_LABELS, ProgressReporter
from app.ai_agents.prompt import (
    format_current_time_prompt,
    language_detection_prompt,
    quick_reply_response_prompt,
    workflow_date_extraction_prompt,
    workflow_intent_classification_prompt,
    workflow_intent_classification_with_pending_create_event_prompt,
    workflow_intent_classification_with_pending_create_reminder_prompt,
    workflow_intent_classification_with_pending_generic_prompt,
    workflow_system_prompt,
    workflow_update_pending_action_prompt,
)
from app.ai_agents.schemas.workflow_schemas import RealValueContext
from app.ai_agents.token_tracking import (
    TokenUsageContext,
    reset_token_usage_context,
    set_token_usage_context,
)
from app.ai_agents.tools import (
    create_balance_tools,
    create_calendar_tools,
    create_company_info_tools,
    create_email_tools,
    create_finance_tools,
    create_health_tools,
    create_productivity_tools,
    create_profile_tools,
    create_reminder_tools,
    create_search_tools,
    create_smart_suggestions_tools,
)
from app.ai_agents.tools.executor.reminder_tool_utils import localize_priority
from app.ai_agents.tools.validate.pending_action_manager import PendingActionManager
from app.ai_agents.utils import format_user_data_as_markdown
from app.ai_agents.utils.formatters import compact_user_data
from app.ai_agents.utils.time_utils import resolve_date_expression
from app.config import settings
from app.services.executor.memory_service_impl import MemoryServiceImpl
from app.utils.logger import logger

# ============================================================================
# STATE DEFINITION
# ============================================================================


class HybridState(TypedDict):
    """State for the hybrid workflow."""

    # Input
    user_message: str
    chat_history: List[Dict[str, str]]
    user_id: str
    session_id: str
    request_id: str
    app_id: str
    bot_name: str
    user_data: Dict[str, Any]
    timezone: str
    quick_reply: Optional[Dict[str, str]]

    # Language detection
    language_code: str
    language_name: str
    last_language_code: str

    # Intent classification
    intent: Optional[Dict[str, Any]]

    # Real value context
    real_values: Optional[Dict[str, Any]]

    # Smart suggestions hand-off (captured from the smart_suggestions tool in
    # execute_agent; routes plan-mode suggestions to the multi-action pipeline)
    smart_suggestions_data: Optional[Dict[str, Any]]

    # Agent execution
    messages: List[Any]
    agent_response: str
    tool_results: List[Dict[str, Any]]

    # Quick reply result
    quick_reply_result: Optional[Dict[str, Any]]

    # Pending action context
    pending_action_context: Optional[Dict[str, Any]]

    # Output
    response: str
    extracted_info: Optional[Dict[str, Any]]
    quick_reply_buttons: Optional[List[Dict[str, str]]]
    error: Optional[str]
    action_completed: Optional[str]
    # Date expressions (resolved from natural language, e.g. "tomorrow", "next Monday")
    date_expressions: Optional[List[Dict[str, Any]]]
    action_queue: Optional[List[Dict[str, Any]]]
    complete_action_queue: Optional[List[Dict[str, Any]]]
    multi_action_mode: bool
    has_pending_confirmation: Optional[bool]
    auto_confirm_sync_available: bool
    auto_confirm_sync_event_count: int
    # Deferred LTM read/write asyncio.Task handles. Ephemeral — not checkpoint-safe;
    # move to a real node / out of state if a checkpointer is added (same constraint for both).
    # Read is awaited in merge_preprocess; write is kicked off there after retrieval so
    # known_memory reaches ingest. Mutation turns await the write before agent prompt.
    ltm_read_task: Optional[Any]
    # Quick replies carry no new user facts — skip the LTM write for those turns.
    skip_ltm_write: bool
    ltm_write_task: Optional[Any]


# ============================================================================
# HYBRID WORKFLOW
# ============================================================================


class HybridWorkflow:
    """
    Hybrid Workflow combining Agentic Workflow + ReAct Agent.

    Flow:
    1. DETECT LANGUAGE (AI)
    2. CLASSIFY INTENT (AI + Structured Output)
    3. GATHER CONSTRAINTS (if needed - get real IDs)
    4. EXECUTE AGENT (ReAct with constraints)
    5. VALIDATE OUTPUT (AI)
    6. GENERATE RESPONSE
    """

    def __init__(self):
        """Initialize the hybrid workflow."""
        llm_manager = LLMManager()
        self.llm = llm_manager.create_tracked_model(temperature=0.1)
        self.suggestions_llm = llm_manager.create_tracked_model(
            temperature=0.1, reasoning_effort="high"
        )
        self.validator_llm = llm_manager.create_tracked_model(temperature=0)
        self.memory_runtime = get_memory_runtime_service()
        self.memory_service = MemoryServiceImpl()
        self.graph = self._build_graph()
        # Strong refs to in-flight fire-and-forget LTM writes so the loop doesn't GC them.
        self._bg_writes: set = set()

    def _start_background_write(
        self,
        *,
        user_id: str,
        session_id: str,
        message: str,
        language_code: str,
        user_data: Dict[str, Any],
        chat_history: List[Dict[str, str]],
    ) -> asyncio.Task[LTMWriteResult]:
        """Persist this turn's LTM facts; returns a task with a gate barrier.

        The reply doesn't need the write to finish for normal storable-fact turns;
        only memory mutations (corrections/deletes) are awaited before prompt build.
        Tracked in ``self._bg_writes`` so the loop doesn't GC it.
        """
        gate_ready = asyncio.Event()

        async def _write() -> LTMWriteResult:
            is_mutation = False
            try:
                window = MemoryConstants.LTM_EXTRACTION_WINDOW_MESSAGES
                recent_messages = (
                    chat_history[-window:] if chat_history and window > 0 else None
                )
                gate_decision, extracted_signals = (
                    await self.memory_runtime.signal_extractor.evaluate_ingest_turn(
                        user_message=message,
                        language_code=language_code or "en",
                        user_data=user_data,
                        recent_messages=recent_messages,
                    )
                )
                is_mutation = bool(
                    gate_decision and gate_decision.is_memory_mutation
                )
                # Signal the gate barrier *before* append so non-mutation turns do not
                # block the agent critical path on the full write.
                write_task.is_memory_mutation = is_mutation
                write_task.gate_ready.set()
                summary = await self.memory_runtime.append_interaction_events(
                    user_id=user_id or "",
                    session_id=session_id or "",
                    user_message=message,
                    language_code=language_code or "en",
                    user_data=user_data,
                    chat_history=chat_history,
                    gate_decision=gate_decision,
                    extracted_signals=extracted_signals,
                )
                return LTMWriteResult(
                    is_memory_mutation=is_mutation,
                    summary=summary,
                )
            except Exception as exc:
                logger.warning(f"LTM background write failed softly: {exc}")
                return LTMWriteResult(is_memory_mutation=is_mutation, summary=None)
            finally:
                # Safety net if gate eval failed before the early set above.
                write_task.gate_ready.set()

        write_task = asyncio.ensure_future(_write())
        write_task.gate_ready = gate_ready
        write_task.is_memory_mutation = False
        self._bg_writes.add(write_task)
        write_task.add_done_callback(self._on_bg_write_done)
        return write_task

    async def _await_ltm_write_for_agent(self, state: HybridState) -> Dict[str, Any]:
        """Gate-barrier: await mutation writes before building the agent prompt."""
        write_task = state.get("ltm_write_task")
        if not write_task:
            return {}

        user_data = dict(state.get("user_data") or {})
        try:
            await write_task.gate_ready.wait()
            if not write_task.is_memory_mutation:
                return {}
            result = await write_task
            if result.summary:
                user_data["ltm_write_summary"] = result.summary
                return {"user_data": user_data}
        except Exception as exc:
            logger.warning(f"LTM mutation write await failed softly: {exc}")
        return {}

    def _on_bg_write_done(self, task: Any) -> None:
        """Drop the strong ref and surface (retrieve) any error so asyncio stays quiet."""
        self._bg_writes.discard(task)
        if not task.cancelled():
            exc = task.exception()
            if exc is not None:
                logger.warning(f"LTM background write failed softly: {exc}")

    def _debug_truncate(self, value: Any, limit: int = 280) -> str:
        text = " ".join(str(value or "").split())
        if len(text) <= limit:
            return text
        return text[:limit].rstrip() + "..."

    async def _emit_progress(self, state: Dict[str, Any], step_id: str) -> None:
        """Write a status="process" step label to the shared response key."""
        try:
            request_id = state.get("request_id")
            user_id = state.get("user_id")
            if not (request_id and user_id):
                return
            lang = (
                state.get("language_code") or state.get("last_language_code") or "en"
            ).lower()
            labels = PROGRESS_LABELS[step_id]
            label = labels.get(lang) or labels["en"]
            key = self.memory_service.get_response_key(
                user_id, state.get("session_id", ""), request_id
            )
            await self.memory_service.set_json(
                key,
                {"status": "process", "step": step_id, "label": label},
                ttl=settings.response_cache_ttl,
            )
        except Exception as e:
            logger.debug(f"progress emit skipped ({step_id}): {e}")

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(HybridState)

        # Add nodes
        workflow.add_node("check_quick_reply", self._check_quick_reply_node)
        workflow.add_node("detect_language", self._detect_language_node)
        workflow.add_node("check_pending_action", self._check_pending_action_node)
        workflow.add_node("classify_intent", self._classify_intent_node)
        workflow.add_node(
            "extract_date_expressions", self._extract_date_expressions_node
        )
        workflow.add_node("clear_pending_action", self._clear_pending_action_node)
        workflow.add_node("execute_agent", self._execute_agent_node)
        workflow.add_node("execute_update_agent", self._execute_update_agent_node)
        workflow.add_node("validate_output", self._validate_output_node)
        workflow.add_node("emit_suggestion", self._emit_suggestion_node)
        workflow.add_node("generate_response", self._generate_response_node)
        workflow.add_node("extract_multi_actions", self._extract_multi_actions_node)
        workflow.add_node("execute_single_action", self._execute_single_action_node)
        workflow.add_node("summarize_multi_actions", self._summarize_multi_actions_node)
        workflow.add_node("merge_preprocess", self._merge_preprocess_node)

        # Define edges
        workflow.set_entry_point("check_quick_reply")
        workflow.add_conditional_edges(
            "check_quick_reply",
            self._route_quick_reply,
            {
                "generate_response": "generate_response",  # If quick_reply exists, generate response
                "execute_single_action": "execute_single_action",
                "continue_flow": "check_pending_action",  # Free-text → normal flow
            },
        )
        # Fan-out: language, intent, and date extraction run in parallel. Each reads
        # only message + pending_action_context (+ timezone) and writes a disjoint
        # state key (language_code/name | intent | date_expressions), so LangGraph
        # merges them without a reducer. The date node self-gates (always runs,
        # returns [] when there is no date) so it no longer depends on intent.
        workflow.add_edge("check_pending_action", "detect_language")
        workflow.add_edge("check_pending_action", "classify_intent")
        workflow.add_edge("check_pending_action", "extract_date_expressions")

        # Barrier: wait for all three before routing.
        workflow.add_edge("detect_language", "merge_preprocess")
        workflow.add_edge("classify_intent", "merge_preprocess")
        workflow.add_edge("extract_date_expressions", "merge_preprocess")

        # Routing depends only on intent now (multi-action / pending); date is already
        # resolved in state so there is no has_date_expression branch.
        workflow.add_conditional_edges(
            "merge_preprocess",
            self._route_after_preprocess,
            {
                "extract_multi_actions": "extract_multi_actions",
                "execute_update_agent": "execute_update_agent",
                "clear_pending_action": "clear_pending_action",
                "execute_agent": "execute_agent",
            },
        )
        workflow.add_conditional_edges(
            "extract_multi_actions",
            self._route_after_extract_multi,
            {
                "generate_response": "generate_response",
                "execute_single_action": "execute_single_action",
            },
        )
        workflow.add_conditional_edges(
            "execute_single_action",
            self._route_after_single_action,
            {
                "execute_single_action": "execute_single_action",
                "summarize_multi_actions": "summarize_multi_actions",
                "generate_response": "generate_response",
            },
        )
        workflow.add_edge("summarize_multi_actions", "generate_response")
        workflow.add_edge("clear_pending_action", "execute_agent")
        workflow.add_edge("execute_update_agent", "validate_output")
        # execute_agent is now a router: plan-mode smart_suggestions hand off to
        # the multi-action pipeline; everything else falls through to
        # validate_output exactly as before.
        workflow.add_conditional_edges(
            "execute_agent",
            self._route_after_agent,
            {
                "extract_multi_actions": "extract_multi_actions",
                "emit_suggestion": "emit_suggestion",
                "validate_output": "validate_output",
            },
        )
        workflow.add_edge("emit_suggestion", "generate_response")
        workflow.add_edge("validate_output", "generate_response")
        workflow.add_edge("generate_response", END)

        return workflow.compile()

    # ─────────────────────────────────────────────────────────────
    # ROUTING FUNCTIONS
    # ─────────────────────────────────────────────────────────────

    def _route_after_agent(self, state: HybridState) -> str:
        """Route after the main ReAct agent.

        Default is ``validate_output`` (unchanged behaviour). Only when the
        agent produced a plan-mode smart_suggestions result do we hand off to
        ``extract_multi_actions`` so the suggested actions run through the
        multi-action pipeline (plan → confirm → sequential execute).
        """
        ss = state.get("smart_suggestions_data")
        if ss and ss.get("mode") == "plan" and ss.get("suggestions"):
            logger.info(
                f"[SS-HANDOFF] step 3 | route execute_agent -> "
                f"extract_multi_actions ({len(ss['suggestions'])} suggestion(s))"
            )
            return "extract_multi_actions"
        # Deterministic emit: a ready `message` (improve / evaluate) is delivered
        # verbatim by code — the main agent never re-renders it. followup/error/
        # empty envelopes carry no message and still fall through to the agent.
        if ss and (ss.get("message") or "").strip() and not ss.get("followup_kind"):
            logger.info("[SS-EMIT] ready message -> emit_suggestion (verbatim)")
            return "emit_suggestion"
        return "validate_output"

    def _route_quick_reply(self, state: HybridState) -> str:
        """Route based on quick_reply processing result."""
        if state.get("multi_action_mode") and not state.get("quick_reply_result"):
            logger.info("🔀 Multi-action resuming → execute_single_action")
            return "execute_single_action"

        if state.get("quick_reply_result"):
            logger.info("🔘 Quick reply processed → generate_response")
            return "generate_response"

        return "continue_flow"

    def _route_by_intent(self, state: HybridState) -> str:
        """Route based on classified intent."""
        # Always route to execute_agent
        # Real IDs will be gathered by middleware when agent calls list_events/get_today_events
        logger.info("🚀 Routing to execute_agent")
        return "execute_agent"

    def _route_by_pending_action(self, state: HybridState) -> str:
        """Route based on pending action and intent classification."""
        pending_context = state.get("pending_action_context", {})
        intent = state.get("intent", {})

        # No pending action → normal flow
        if not pending_context.get("has_pending"):
            logger.info("🚀 Routing to execute_agent (normal flow)")
            return "execute_agent"

        # Has pending action
        is_update = intent.get("is_update_to_pending", False)
        is_new_intent = intent.get("is_new_intent", False)

        if is_update and not is_new_intent:
            # User wants to update pending action
            logger.info("🔄 Routing to execute_update_agent (updating pending action)")
            return "execute_update_agent"
        elif is_new_intent:
            # User wants new intent → clear pending and continue normal flow
            logger.info("🆕 Routing to clear_pending_action (new intent detected)")
            return "clear_pending_action"
        else:
            # Default: continue normal flow (fallback)
            logger.info("🚀 Routing to execute_agent (normal flow - fallback)")
            return "execute_agent"

    def _route_after_preprocess(self, state: HybridState) -> str:
        """Route after the parallel preprocess barrier (language ∥ intent ∥ date).

        date_expressions is already resolved in state (the date node self-gates and
        always runs), so routing depends only on intent: multi-action first, then the
        pending-action decision. There is no has_date_expression branch anymore.
        """
        intent = state.get("intent") or {}

        if intent.get("is_multi_action", False):
            logger.info("🔀 Routing to extract_multi_actions")
            return "extract_multi_actions"

        return self._route_by_pending_action(state)

    def _route_after_extract_multi(self, state: HybridState) -> str:
        if state.get("multi_action_mode") and state.get("response"):
            logger.info("🔀 Show plan → generate_response")
            return "generate_response"

        logger.info("🔀 Fallback/Skip plan → execute_single_action")
        return "execute_single_action"

    async def _merge_preprocess_node(self, state: HybridState) -> Dict[str, Any]:
        """Barrier for the language/intent/date fan-out; also awaits the deferred LTM read,
        merges recalled context into user_data, and kicks off the fire-and-forget write so
        known_memory from retrieval reaches ingest_interaction.
        ltm_read_task holds an asyncio.Task — move to a real node if a checkpointer is added.
        """
        task = state.get("ltm_read_task")
        if not task:
            return {}

        user_id = state.get("user_id", "")
        session_id = state.get("session_id", "")
        try:
            retrieval_result = await task
        except Exception as e:
            logger.warning(
                f"Failed to retrieve long-term memory context for {user_id}: {e}"
            )
            return {}

        user_data = {**state.get("user_data", {})}
        user_data["ltm_retrieval_result"] = retrieval_result
        item_count = len(getattr(retrieval_result, "items", None) or [])
        logger.info(
            "LTM retrieval outcome=retrieval_result "
            f"user_id={user_id} session_id={session_id} "
            f"candidate_count={retrieval_result.candidate_count} "
            f"memory_types={retrieval_result.memory_types} "
            f"item_count={item_count}"
        )

        if not state.get("skip_ltm_write"):
            ltm_write_task = self._start_background_write(
                user_id=user_id,
                session_id=session_id,
                message=state.get("user_message", ""),
                language_code=state.get("last_language_code")
                or state.get("language_code")
                or "en",
                user_data=user_data,
                chat_history=state.get("chat_history") or [],
            )
        else:
            ltm_write_task = None

        return {"user_data": user_data, "ltm_write_task": ltm_write_task}

    # ─────────────────────────────────────────────────────────────
    # NODE 0: CHECK QUICK REPLY
    # ─────────────────────────────────────────────────────────────

    async def _check_quick_reply_node(self, state: HybridState) -> Dict[str, Any]:
        """Check if this is a quick_reply action and handle it."""
        quick_reply = state.get("quick_reply")

        if not quick_reply or not quick_reply.get("value"):
            # No quick_reply, continue normal flow
            return {}

        quick_reply_value = quick_reply.get("value", "")
        logger.info(f"🔘 Processing quick_reply: {quick_reply_value}")

        MULTI_ACTION_VALUES = (
            QuickReply.MULTI_ACTION_CONTINUE,
            QuickReply.MULTI_ACTION_CANCEL,
        )
        if quick_reply_value in MULTI_ACTION_VALUES:
            return await self._handle_multi_action_quick_reply(state, quick_reply)

        # Check if language_code is provided in quick_reply
        result_updates = {}
        if quick_reply.get("language_code"):
            language_code = quick_reply.get("language_code")
            language_name = LANGUAGE_NAMES.get(language_code, "English")
            result_updates["language_code"] = language_code
            result_updates["language_name"] = language_name
            logger.info(
                f"🌐 Language set from quick_reply: {language_name} ({language_code})"
            )

        try:
            user_id = state.get("user_id", "")
            session_id = state.get("session_id", "")
            user_data = state.get("user_data", {})
            timezone = state.get("timezone")

            # Create service and handle quick reply (async)
            # Pass timezone from state to ensure calendar events use correct timezone
            pending_action_service = PendingActionService(
                user_id, session_id, user_data, timezone=timezone
            )
            quick_reply_result = await pending_action_service.handle_quick_reply(
                quick_reply
            )
            result_updates["quick_reply_result"] = quick_reply_result
            return result_updates

        except Exception as e:
            logger.exception(f"Error handling quick_reply: {e}")
            result_updates["quick_reply_result"] = {
                "status": ToolStatus.ERROR,
                "error": str(e),
            }
            return result_updates

    async def _handle_multi_action_quick_reply(
        self,
        state: HybridState,
        quick_reply: Dict[str, str],
    ) -> Dict[str, Any]:
        return await handle_multi_action_quick_reply(
            state=state,
            quick_reply=quick_reply,
            llm=self.llm,
            memory_service=self.memory_service,
            extract_text_fn=self._extract_text_from_content,
            language_names=LANGUAGE_NAMES,
        )

    async def _process_multi_action_cancel(
        self,
        state: HybridState,
        quick_reply: Dict[str, str],
    ) -> Dict[str, Any]:
        return await process_multi_action_cancel(
            state=state,
            quick_reply=quick_reply,
            llm=self.llm,
            memory_service=self.memory_service,
            extract_text_fn=self._extract_text_from_content,
            language_names=LANGUAGE_NAMES,
        )

    async def _process_multi_action_continue(
        self,
        state: HybridState,
        quick_reply: Dict[str, str],
    ) -> Dict[str, Any]:
        return await process_multi_action_continue(
            state=state,
            quick_reply=quick_reply,
            memory_service=self.memory_service,
            language_names=LANGUAGE_NAMES,
        )

    # ─────────────────────────────────────────────────────────────
    # NODE 1: DETECT LANGUAGE (AI-driven)
    # ─────────────────────────────────────────────────────────────

    async def _detect_language_node(self, state: HybridState) -> Dict[str, Any]:
        """Detect user's language using AI."""
        logger.info("🌐 Step 1: Detecting language...")

        message = state["user_message"]
        last_language_code = state["last_language_code"]
        last_language = LANGUAGE_NAMES.get(last_language_code, "")
        pending_context = state.get("pending_action_context", {})
        has_pending = pending_context.get("has_pending", False)

        keep_last_language = self._should_keep_last_language(
            message=message,
            last_language_code=last_language_code,
            last_language=last_language,
            has_pending=has_pending,
        )
        if keep_last_language:
            language_code = last_language_code
            language_name = last_language
        else:
            try:
                prompt = language_detection_prompt(
                    message, last_language_code, last_language
                )
                response = await self.validator_llm.ainvoke(
                    [HumanMessage(content=prompt)]
                )
                # Extract text from content (may be list with reasoning_content)
                content_text = self._extract_text_from_content(response.content)
                data = self._parse_json_response(content_text)

                language_code = data.get("language_code", "en")
                language_name = data.get("language_name", "English")
                confidence = data.get("confidence", 0.0)
                logger.info(f"   → Detected: {language_name} ({confidence:.2f})")
                if float(confidence) < 0.85:
                    logger.warning(
                        f"Language detection confidence is too low: {confidence}, defaulting to Last Language Code: {last_language_code}"
                    )
                    language_code = (
                        last_language_code if last_language_code else "Other"
                    )
                    language_name = LANGUAGE_NAMES.get(language_code, "")
            except Exception as e:
                logger.warning(
                    f"Language detection failed: {e}, defaulting to Last Language Code"
                )
                language_code = last_language_code if last_language_code else "en"
                language_name = LANGUAGE_NAMES.get(last_language_code, "English")

        updates = {"language_code": language_code, "language_name": language_name}
        # First progress label emitted here (not before the graph) so it uses the
        # freshly detected language, not the previous turn's last_language_code.
        await self._emit_progress({**state, **updates}, "analyzing")
        return updates

    # ─────────────────────────────────────────────────────────────
    # NODE 2: CHECK PENDING ACTION
    # ─────────────────────────────────────────────────────────────

    def _check_pending_action_node(self, state: HybridState) -> Dict[str, Any]:
        """Check for pending actions in cache (for calendar tools)."""
        logger.info("🔍 Step 2: Checking pending actions...")

        user_id = state.get("user_id")
        session_id = state.get("session_id", "")
        if not user_id:
            logger.warning("   → No user_id, skipping pending action check")
            return {
                "pending_action_context": {
                    "has_pending": False,
                    "tool_name": None,
                    "cached_data": None,
                    "summary": None,
                }
            }

        # Tools that might have a pending action (calendar + reminder).
        pending_tools = [
            "create_event_by_name",
            "update_event_by_name",
            "delete_event_by_name",
            "sync_event_by_name",
            "create_reminder",
            "update_reminder",
            "delete_reminder",
            "mark_off_reminder",
        ]
        reminder_tools = {
            "create_reminder",
            "update_reminder",
            "delete_reminder",
            "mark_off_reminder",
        }
        pending_action_manager = PendingActionManager()

        for tool_name in pending_tools:
            cached_data = pending_action_manager.get(user_id, session_id, tool_name)
            # Check for all valid pending statuses
            valid_statuses = (
                ToolStatus.NEED_CONFIRMATION,
                ToolStatus.MISSING,
                ToolStatus.SUGGEST,
                ToolStatus.EDITING,
                ToolStatus.OVERLAPPING,
            )
            if cached_data and cached_data.get("status") in valid_statuses:
                status = cached_data.get("status")
                logger.info(
                    f"   → Found pending action: {tool_name} (status: {status})"
                )

                if tool_name in reminder_tools:
                    summary = self._build_reminder_pending_summary(
                        tool_name, status, cached_data
                    )
                    return {
                        "pending_action_context": {
                            "has_pending": True,
                            "tool_name": tool_name,
                            "cached_data": cached_data,
                            "summary": summary,
                        }
                    }

                # Generate summary for display
                body = cached_data.get("body", {})
                events = body.get("events", []) if isinstance(body, dict) else []
                params = cached_data.get("params", {})
                first_event = events[0] if events else {}
                event_name = first_event.get("summary") or params.get(
                    "summary", "Event"
                )

                if status == ToolStatus.NEED_CONFIRMATION:
                    if events:
                        summary = f"Create {len(events)} event(s): '{event_name}'"
                    else:
                        summary = f"Create event: '{event_name}'"
                elif status == ToolStatus.SUGGEST:
                    suggested_fields = cached_data.get("suggested_fields", [])
                    summary = f"Event '{event_name}' - suggest adding: {', '.join(suggested_fields)}"
                elif status == ToolStatus.EDITING:
                    summary = f"Editing event '{event_name}' - waiting for your changes"
                elif status == ToolStatus.OVERLAPPING:
                    overlapping_events = cached_data.get("overlapping_events", [])
                    overlap_names = [
                        e.get("summary", "Event") for e in overlapping_events
                    ]
                    summary = f"Event '{event_name}' overlaps with: {', '.join(overlap_names)}"
                else:
                    # status == ToolStatus.MISSING
                    missing_fields = cached_data.get("missing_fields", [])
                    summary = f"Creating event '{event_name}' - missing: {', '.join(missing_fields)}"

                return {
                    "pending_action_context": {
                        "has_pending": True,
                        "tool_name": tool_name,
                        "cached_data": cached_data,
                        "summary": summary,
                    }
                }

        logger.info("   → No pending actions found")
        return {
            "pending_action_context": {
                "has_pending": False,
                "tool_name": None,
                "cached_data": None,
                "summary": None,
            }
        }

    def _build_reminder_pending_summary(
        self, tool_name: str, status: str, cached_data: Dict[str, Any]
    ) -> str:
        """Human-readable summary for a pending reminder action (display only)."""
        body = cached_data.get("body", {})
        body = body if isinstance(body, dict) else {}

        if status == ToolStatus.MISSING:
            missing = cached_data.get("missing_fields", [])
            return f"Reminder action pending - missing: {', '.join(missing)}"

        if status == ToolStatus.SUGGEST:
            reminders = body.get("reminders", []) or []
            title = (reminders[0].get("title") if reminders else None) or "Reminder"
            suggested = cached_data.get("suggested_fields", [])
            return f"Reminder '{title}' - suggest adding: {', '.join(suggested)}"

        if tool_name == "create_reminder":
            reminders = body.get("reminders", []) or []
            title = (reminders[0].get("title") if reminders else None) or "Reminder"
            return f"Create {len(reminders) or 1} reminder(s): '{title}'"
        if tool_name == "update_reminder":
            title = body.get("title")
            return f"Update reminder: '{title}'" if title else "Update reminder"
        if tool_name == "delete_reminder":
            ids = body.get("reminderIds", []) or []
            return f"Delete {len(ids) or 1} reminder(s)"
        if tool_name == "mark_off_reminder":
            items = body.get("items", []) or []
            return f"Mark off {len(items) or 1} reminder(s)"
        return "Reminder action - waiting for confirmation"

    def _format_reminder_due(self, due: Any) -> str:
        """Readable due date for the preview (drop raw ISO 'T'/'Z')."""
        if not due:
            return ""
        s = str(due).replace("Z", "").strip()
        if "T" in s:
            date_part, time_part = s.split("T", 1)
            return f"{date_part} {time_part[:5]}".strip()
        return s

    def _build_pending_reminder_preview(
        self,
        user_id: Optional[str],
        session_id: str,
        language_code: Optional[str] = None,
    ) -> str:
        """Deterministic list of every reminder awaiting confirmation ('' if none).

        Appended to the agent's confirmation message so the user always sees each
        item to review, independent of how the LLM phrased its reply (the LLM tends
        to collapse long batches into a count + date range).
        """
        if not user_id:
            return ""
        pam = PendingActionManager()

        create_cached = pam.get(user_id, session_id, "create_reminder")
        if (
            create_cached
            and create_cached.get("status") == ToolStatus.NEED_CONFIRMATION
        ):
            body = create_cached.get("body", {})
            reminders = body.get("reminders", []) if isinstance(body, dict) else []
            if reminders:
                lines = []
                for i, r in enumerate(reminders, 1):
                    title = str(r.get("title") or "Reminder").strip()
                    parts = [f"**{title}**"]
                    due = self._format_reminder_due(r.get("due_date"))
                    if due:
                        parts.append(due)
                    priority = localize_priority(r.get("priority"), language_code)
                    if priority:
                        parts.append(priority)
                    lines.append(f"{i}. " + " — ".join(parts))
                return "\n".join(lines)

        mark_off_cached = pam.get(user_id, session_id, "mark_off_reminder")
        if (
            mark_off_cached
            and mark_off_cached.get("status") == ToolStatus.NEED_CONFIRMATION
        ):
            preview = mark_off_cached.get("preview") or {}
            reminders = (
                preview.get("reminders", []) if isinstance(preview, dict) else []
            )
            if reminders:
                lines = []
                for i, r in enumerate(reminders, 1):
                    title = str(r.get("title") or "Reminder").strip()
                    lines.append(f"{i}. **{title}**")
                return "\n".join(lines)

        return ""

    # ─────────────────────────────────────────────────────────────
    # NODE 3: CLASSIFY INTENT (AI + Structured Output)
    # ─────────────────────────────────────────────────────────────

    async def _classify_intent_node(self, state: HybridState) -> Dict[str, Any]:
        """Classify user intent using AI with structured output."""
        logger.info("🎯 Step 3: Classifying intent...")

        message = state["user_message"]
        chat_history = state.get("chat_history", [])

        # Check if there's a pending action
        pending_context = state.get("pending_action_context", {})
        has_pending = pending_context.get("has_pending", False)

        # Find last assistant message from chat history
        last_assistant_msg = None
        for msg in reversed(chat_history):
            if msg.get("role") == "assistant":
                last_assistant_msg = msg.get("content", "")
                break

        # Add AI message if found in history
        if last_assistant_msg:
            logger.info(
                f"   → Using last AI message from history for intent classification node"
            )

        if has_pending:
            # Use pending action context prompt
            pending_tool_name = pending_context.get("tool_name")
            pending_summary = pending_context.get("summary", "")
            cached_data = pending_context.get("cached_data", {})

            logger.info(
                f"   → Using pending action context prompt for: {pending_tool_name} with summary: {pending_summary}"
            )

            # Select appropriate prompt based on tool name
            if pending_tool_name == "create_event_by_name":
                prompt = (
                    workflow_intent_classification_with_pending_create_event_prompt(
                        message=message,
                        last_message=last_assistant_msg if last_assistant_msg else "",
                        pending_action_summary=pending_summary,
                        cached_data=cached_data,
                    )
                )
            elif pending_tool_name == "create_reminder":
                prompt = (
                    workflow_intent_classification_with_pending_create_reminder_prompt(
                        message=message,
                        last_message=last_assistant_msg if last_assistant_msg else "",
                        pending_action_summary=pending_summary,
                        cached_data=cached_data,
                    )
                )
            else:
                # Fallback to generic prompt
                prompt = workflow_intent_classification_with_pending_generic_prompt(
                    message=message,
                    last_message=last_assistant_msg if last_assistant_msg else "",
                    pending_action_summary=pending_summary,
                    pending_tool_name=pending_tool_name or "unknown",
                )
        else:
            # Normal intent classification
            prompt = workflow_intent_classification_prompt(
                message, last_assistant_msg if last_assistant_msg else ""
            )

        # Build conversation pair: Get last AI message from history if available
        messages = []
        # Add classification prompt as user message
        messages.append(HumanMessage(content=prompt))
        try:
            response = await self.validator_llm.ainvoke(messages)
            # Extract text from content (may be list with reasoning_content)
            content_text = self._extract_text_from_content(response.content)
            data = self._parse_json_response(content_text)
            # Map string values to enum values
            category = data.get("category", "chat")
            action = data.get("action", "chat")
            complexity = data.get("complexity", "simple")
            has_date_expression = data.get("has_date_expression", False)

            logger.info(f"   → Category: {category}")
            logger.info(f"   → Action: {action}")
            logger.info(f"   → Complexity: {complexity}")
            logger.info(f"   → Needs real IDs: {data.get('needs_real_ids', False)}")
            logger.info(f"   → Has date expression: {bool(has_date_expression)}")

            return {
                "intent": {
                    "category": category,
                    "action": action,
                    "complexity": complexity,
                    "needs_real_ids": data.get("needs_real_ids", False),
                    "needs_confirmation": data.get("needs_confirmation", False),
                    "extracted_entities": data.get("extracted_entities", {}),
                    "summary": data.get("summary", message[:100]),
                    "has_date_expression": bool(has_date_expression),
                    # Pending action specific fields
                    "is_update_to_pending": data.get("is_update_to_pending", False),
                    "is_new_intent": data.get("is_new_intent", False),
                    "is_multi_action": data.get("is_multi_action", False),
                }
            }

        except Exception as e:
            logger.warning(f"Intent classification failed: {e}")
            # Default to chat
            return {
                "intent": {
                    "category": "chat",
                    "action": "chat",
                    "complexity": "simple",
                    "needs_real_ids": False,
                    "needs_confirmation": False,
                    "extracted_entities": {},
                    "summary": message[:100],
                    "is_update_to_pending": False,
                    "is_new_intent": False,
                    "has_date_expression": False,
                }
            }

    # ─────────────────────────────────────────────────────────────
    # NODE: EXTRACT DATE EXPRESSIONS
    # ─────────────────────────────────────────────────────────────

    async def _extract_date_expressions_node(
        self, state: HybridState
    ) -> Dict[str, Any]:
        """Extract and resolve natural language date expressions from the user message.

        Self-gating: this node always runs (in parallel with language + intent) and
        returns [] when the message has no date, so it does not depend on intent's
        has_date_expression flag.
        """
        logger.info("📅 Extracting date expressions from message...")

        message = state.get("user_message", "")
        if not message:
            return {"date_expressions": []}

        # Runs in parallel with detect_language, so language_name is not set yet. Use
        # last_language_code (stable within a conversation) as a best-effort locale
        # hint; the prompt overrides it from the message content when they disagree.
        lang_code = state.get("last_language_code") or "en"
        language = LANGUAGE_NAMES.get(lang_code, "English")
        timezone = state.get("timezone")

        current_time_str = format_current_time_prompt(timezone)

        prompt = workflow_date_extraction_prompt(
            message=message, language=language, current_time=current_time_str
        )

        try:
            response = await self.validator_llm.ainvoke([HumanMessage(content=prompt)])
            content_text = self._extract_text_from_content(response.content)
            data = self._parse_json_response(content_text)

            expressions = data.get("expressions") or []
            resolved = resolve_date_expression(expressions, timezone)
            logger.info(f"   → Resolved {len(resolved)} date expressions")

            return {"date_expressions": resolved}

        except Exception as e:
            logger.warning(f"Date expression extraction failed: {e}")
            return {"date_expressions": []}

    async def _extract_multi_actions_node(self, state: HybridState) -> Dict[str, Any]:
        await self._emit_progress(state, "planning")
        logger.info("🔀 Extracting multi-actions...")

        user_id = state.get("user_id", "")
        session_id = state.get("session_id", "")
        user_data = state.get("user_data", {})
        timezone = state.get("timezone")
        language = state.get("language_name", "English")
        language_code = state.get("language_code", "en")
        user_message = state.get("user_message", "")
        messages = state.get("messages", [])

        # ── Hand-off from smart_suggestions (plan mode) ──────────────────
        # Build the draft + advisory directly from plan suggestions (no extra LLM
        # extract). None → no executable suggestions → normal LLM extract below.
        handoff = await build_suggestion_handoff(
            state,
            llm=self.llm,
            memory_service=self.memory_service,
            extract_text_fn=self._extract_text_from_content,
            fallback_fn=self._get_fallback_response,
            create_buttons_fn=self._create_quick_reply_buttons,
        )
        if handoff is not None:
            return handoff

        recent_messages = [msg for msg in messages if msg.type in ("human", "ai")][-4:]
        chat_history_str = "\n".join(
            [
                f"{'User' if msg.type == 'human' else 'Bot'}: {msg.content}"
                for msg in recent_messages
            ]
        )

        date_expressions = state.get("date_expressions") or []

        resolved_dates_section = self._format_resolved_date_expressions(
            date_expressions
        )

        all_tools = _build_tools(
            user_id,
            session_id,
            user_data,
            timezone,
            language,
            state,
            self.llm,
            suggestions_llm=self.suggestions_llm,
        )
        available_tools_description = "\n".join(
            f"- {t.name}: {getattr(t, 'description', '')}" for t in all_tools
        )

        result = extract_multi_actions(
            user_message=user_message,
            intent=state.get("intent") or {},
            llm_invoke=self.validator_llm.invoke,
            extract_text_fn=self._extract_text_from_content,
            parse_json_fn=self._parse_json_response,
            available_tools_description=available_tools_description,
            current_time=format_current_time_prompt(timezone),
            resolved_dates_section=resolved_dates_section,
            chat_history_str=chat_history_str,
        )

        if result.get("multi_action_mode"):
            action_queue = result["action_queue"]
            plan_text = await plan_multi_actions(
                user_id=user_id,
                session_id=session_id,
                user_message=user_message,
                action_queue=action_queue,
                llm=self.llm,
                memory_service=self.memory_service,
                extract_text_fn=self._extract_text_from_content,
                language_name=language,
            )
            buttons = self._create_quick_reply_buttons(
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

            return {
                "multi_action_mode": True,
                "action_queue": action_queue,
                "complete_action_queue": [],
                "response": plan_text,
                "quick_reply_buttons": buttons,
            }

        return result

    # ─────────────────────────────────────────────────────────────
    # NODE 3: EXECUTE AGENT (ReAct with constraints)
    # ─────────────────────────────────────────────────────────────

    async def _execute_single_action_node(self, state: HybridState) -> Dict[str, Any]:
        await self._emit_progress(state, "executing")
        result = await execute_single_action(
            state=state,
            llm=self.llm,
            validator_llm=self.validator_llm,
            memory_service=self.memory_service,
            extract_text_fn=self._extract_text_from_content,
        )

        action_queue = result.get("action_queue", [])
        language_code = state.get("language_code", "en")

        if action_queue and not result.get("has_pending_confirmation"):
            buttons = self._create_quick_reply_buttons(
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
            result["quick_reply_buttons"] = buttons

        return result

    def _route_after_single_action(self, state: HybridState) -> str:
        if state.get("action_queue") or state.get("has_pending_confirmation"):
            logger.info(
                "🔀 Pausing execution — queue non-empty or pending confirmation"
            )
            return "generate_response"
        return "summarize_multi_actions"

    async def _summarize_multi_actions_node(self, state: HybridState) -> Dict[str, Any]:
        return await summarize_multi_actions(
            state=state,
            llm=self.llm,
            memory_service=self.memory_service,
            extract_text_fn=self._extract_text_from_content,
        )

    async def _execute_agent_node(self, state: HybridState) -> Dict[str, Any]:
        """Execute ReAct agent with unified and equal tools."""
        await self._emit_progress(state, "processing")
        user_id = state.get("user_id")
        session_id = state.get("session_id", "")
        user_data = state.get("user_data", {})
        write_updates = await self._await_ltm_write_for_agent(state)
        if write_updates.get("user_data"):
            user_data = {**user_data, **write_updates["user_data"]}
        bot_name = state.get("bot_name", BotConfig.DEFAULT_BOT_NAME)
        timezone = state.get("timezone")
        language = state.get("language_name", "English")
        chat_history = state.get("chat_history", [])
        intent = state.get("intent", {})
        needs_real_ids = intent.get("needs_real_ids", False)

        logger.info(f"🤖 Executing Unified ReAct Agent Workflow...")

        real_value_context = RealValueContext()

        # ── Initialize tools identically ───────────────────────────
        tools = []
        if user_id:
            tools.extend(create_health_tools(user_id, session_id, timezone=timezone))
            tools.extend(
                create_calendar_tools(
                    user_id,
                    session_id,
                    user_data=user_data,
                    timezone=timezone,
                    user_message=state["user_message"],
                    llm=self.llm,
                )
            )
            tools.extend(
                create_reminder_tools(
                    user_id,
                    session_id,
                    user_data=user_data,
                    timezone=timezone,
                    user_message=state["user_message"],
                    llm=self.llm,
                    real_value_context=real_value_context,
                )
            )
            tools.extend(
                create_company_info_tools(user_id, session_id, state["user_message"])
            )
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
                    user_id,
                    session_id,
                    user_data=user_data,
                    timezone=timezone,
                    llm=self.llm,
                )
            )
            tools.extend(
                create_smart_suggestions_tools(
                    user_id,
                    session_id,
                    user_data=user_data,
                    timezone=timezone,
                    language_name=language,
                    user_message=state.get("user_message"),
                    bot_name=bot_name,
                    llm=self.suggestions_llm,
                    # Inner router picks scope + target_date from these (agent no longer
                    # forms date args) — same resolved dates the prompt shows other tools.
                    date_expressions=state.get("date_expressions") or [],
                    # Streams per-phase sub-step labels while the (slow) tool runs.
                    progress=ProgressReporter(
                        self.memory_service,
                        user_id,
                        session_id,
                        state.get("request_id", ""),
                        state.get("language_code", "en"),
                    ),
                )
            )
            tools.extend(
                create_search_tools(
                    user_id=user_id,
                    timezone=timezone,
                    language=language,
                    user_data=user_data,
                    llm=self.llm,
                )
            )

        # ── Middleware ────────────────────────────────────────────
        middleware = [
            RealValueContextMiddleware(real_value_context),
            AIToolValidationMiddleware(self.validator_llm, real_value_context),
            AIOutputValidationMiddleware(
                self.validator_llm,
                real_value_context,
                language,
                needs_real_ids,
                timezone,
                bot_name,
            ),
            ReasoningContentFilterMiddleware(),
            ToolChoicesValidationMiddleware(tools=tools),
            ToolBoundaryMiddleware(),
            SuggestionHandoffMiddleware(real_value_context),
            # Inner-most tool wrapper: emits per-tool progress only for calls that
            # survive the validation/handoff middlewares above it.
            ToolProgressMiddleware(
                memory_service=self.memory_service,
                user_id=user_id or "",
                session_id=session_id,
                request_id=state.get("request_id", ""),
                language_code=state.get("language_code", "en"),
            ),
            RetryModelCallMiddleware(),
        ]

        # ── System prompt ─────────────────────────────────────────
        system_prompt = self._build_agent_prompt(
            user_data=user_data,
            timezone=timezone,
            language=language,
            real_value_context=real_value_context,
            date_expressions=state.get("date_expressions") or [],
            bot_name=bot_name,
        )

        # ── Map chat history without any modification ───────────────
        messages = [
            (
                HumanMessage(content=msg["content"])
                if msg["role"] == "user"
                else AIMessage(content=msg["content"])
            )
            for msg in chat_history
        ]
        messages.append(HumanMessage(content=state["user_message"]))

        try:
            # Create and execute agent
            agent = create_agent(
                self.llm,
                tools,
                system_prompt=system_prompt,
                middleware=middleware,
                debug=True,
            )
            result = await agent.ainvoke({"messages": messages})
            result_messages = result.get("messages", [])

            response_text = ""
            if result_messages:
                last_msg = result_messages[-1]
                content = (
                    last_msg.content if hasattr(last_msg, "content") else str(last_msg)
                )
                response_text = self._extract_text_from_content(content)

            # Return pure text back into the graph pipeline execution.
            # smart_suggestions_data is populated by SuggestionHandoffMiddleware
            # when the agent called the smart_suggestions tool; _route_after_agent
            # uses it to hand plan-mode suggestions to the multi-action pipeline.
            response_text = response_text or self._get_fallback_response(language)
            reminder_preview = self._build_pending_reminder_preview(
                state.get("user_id"),
                state.get("session_id", ""),
                state.get("language_code"),
            )
            if reminder_preview:
                response_text = f"{response_text}\n\n{reminder_preview}"

            return {
                "agent_response": response_text,
                "messages": result_messages,
                "smart_suggestions_data": real_value_context.smart_suggestions_data,
            }

        except Exception as e:
            logger.exception(f"Agent execution failed: {e}")
            return {
                "agent_response": self._get_error_response(language),
                "error": str(e),
            }

    # ─────────────────────────────────────────────────────────────
    # NODE: CLEAR PENDING ACTION
    # ─────────────────────────────────────────────────────────────

    async def _clear_pending_action_node(self, state: HybridState) -> Dict[str, Any]:
        """Clear pending action from cache when user wants new intent."""
        logger.info("🗑️ Clearing pending action (new intent detected)")

        pending_context = state.get("pending_action_context", {})
        tool_name = pending_context.get("tool_name")
        user_id = state.get("user_id")
        session_id = state.get("session_id", "")

        if tool_name and user_id:
            pending_action_manager = PendingActionManager()
            pending_action_manager.delete(user_id, session_id, tool_name)
            logger.info(f"✅ Cleared pending action: {tool_name}")

        return {"pending_action_context": {"has_pending": False}}

    # ─────────────────────────────────────────────────────────────
    # NODE: EXECUTE UPDATE AGENT
    # ─────────────────────────────────────────────────────────────

    async def _execute_update_agent_node(self, state: HybridState) -> Dict[str, Any]:
        """Execute agent with limited tools to update pending action."""
        await self._emit_progress(state, "updating")
        logger.info("🔄 Step: Executing update agent for pending action...")

        user_id = state.get("user_id")
        session_id = state.get("session_id", "")
        user_data = state.get("user_data", {})
        write_updates = await self._await_ltm_write_for_agent(state)
        if write_updates.get("user_data"):
            user_data = {**user_data, **write_updates["user_data"]}
        timezone = state.get("timezone")
        language = state.get("language_name", "English")
        chat_history = state.get("chat_history", [])
        bot_name = state.get("bot_name", BotConfig.DEFAULT_BOT_NAME)

        # Get pending action context
        pending_context = state.get("pending_action_context", {})
        pending_tool_name = pending_context.get("tool_name")
        cached_data = pending_context.get("cached_data", {})
        pending_summary = pending_context.get("summary", "")

        # Build real value context — starts empty; seeded below for update/mark_off.
        real_value_context = RealValueContext()

        # For a pending update_reminder, seed the original booleans from the cached
        # body so re-editing the pending action does not drop has_time/completed
        # (Spring resets omitted boolean primitives to false on a partial PUT).
        if pending_tool_name == "update_reminder" and isinstance(cached_data, dict):
            cached_body = cached_data.get("body", {})
            reminder_id = cached_data.get("reminder_id") or cached_data.get(
                "params", {}
            ).get("reminder_id")
            if reminder_id and isinstance(cached_body, dict):
                real_value_context.reminder_id_to_details[reminder_id] = {
                    "has_time": cached_body.get("has_time"),
                    "completed": cached_body.get("completed"),
                }

        # Pending mark_off_reminder only has eligible (incomplete) ids in preview;
        # seed title + completed=False so re-validation does not require get_reminders.
        # Also restore dueDateUtc from the cached body items so re-validation can
        # rebuild originalDueDate — without it, recurring occurrences would be
        # dropped and silently skipped by the bulk-tick-off endpoint.
        if pending_tool_name == "mark_off_reminder" and isinstance(cached_data, dict):
            preview = cached_data.get("preview") or {}
            reminders = (
                preview.get("reminders", []) if isinstance(preview, dict) else []
            )
            cached_body = cached_data.get("body") or {}
            body_items = (
                cached_body.get("items", []) if isinstance(cached_body, dict) else []
            )
            due_by_id = {
                str(it.get("reminderId")): it.get("originalDueDate")
                for it in body_items
                if isinstance(it, dict) and it.get("reminderId")
            }
            for item in reminders:
                if not isinstance(item, dict):
                    continue
                rid = item.get("id")
                title = item.get("title")
                if rid and title:
                    details = {"title": title, "completed": False}
                    original_due = due_by_id.get(str(rid))
                    if original_due:
                        details["dueDateUtc"] = original_due
                    real_value_context.reminder_id_to_details[str(rid)] = details

        # Create limited tools - only the pending tool, for updating the pending action
        tools = []
        if user_id and pending_tool_name in [
            "create_event_by_name",
            "update_event_by_name",
            "delete_event_by_name",
            "sync_event_by_name",
        ]:
            calendar_tools = create_calendar_tools(
                user_id,
                session_id,
                user_data=user_data,
                timezone=timezone,
                user_message=state["user_message"],
                llm=self.llm,
            )
            # Filter to only the pending tool
            tools = [tool for tool in calendar_tools if tool.name == pending_tool_name]
        elif user_id and pending_tool_name in [
            "create_reminder",
            "update_reminder",
            "delete_reminder",
            "mark_off_reminder",
        ]:
            reminder_tools = create_reminder_tools(
                user_id,
                session_id,
                user_data=user_data,
                timezone=timezone,
                user_message=state["user_message"],
                llm=self.llm,
                real_value_context=real_value_context,
            )
            tools = [tool for tool in reminder_tools if tool.name == pending_tool_name]

        # Create middleware (simpler, no output validation needed for updates)
        middleware = [
            RealValueContextMiddleware(real_value_context),
            AIToolValidationMiddleware(self.validator_llm, real_value_context),
            ReasoningContentFilterMiddleware(),
            ToolBoundaryMiddleware(),
            RetryModelCallMiddleware(),
        ]

        # Build system prompt for update agent
        current_time_str = format_current_time_prompt(timezone)
        merged_context = build_merged_user_context(
            user_data,
            user_data.get("ltm_retrieval_result"),
            base_profile_markdown=self._format_user_profile(user_data),
        )
        user_profile = merged_context.merged_profile_markdown
        write_summary = user_data.get("ltm_write_summary")
        if write_summary:
            user_profile = f"{user_profile}\n\n{write_summary}".strip()
        system_prompt = workflow_update_pending_action_prompt(
            pending_tool_name=pending_tool_name,
            pending_summary=pending_summary,
            pending_params=cached_data.get("params", {}),
            language=language,
            current_time=current_time_str,
            timezone=timezone or "UTC",
            bot_name=bot_name,
            is_multi_action=state.get("multi_action_mode", False),
            user_profile=user_profile,
        )

        # Build messages
        messages = []
        for msg in chat_history:
            if msg.get("role") == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg.get("role") == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        messages.append(HumanMessage(content=state["user_message"]))

        try:
            # Create and execute agent
            agent = create_agent(
                self.llm, tools, system_prompt=system_prompt, middleware=middleware
            )

            result = await agent.ainvoke({"messages": messages})

            # Extract response
            result_messages = result.get("messages", [])
            response_text = ""

            if result_messages:
                last_msg = result_messages[-1]
                content = None

                if hasattr(last_msg, "content"):
                    content = last_msg.content
                elif isinstance(last_msg, dict):
                    content = last_msg.get("content", "")
                else:
                    content = str(last_msg)

                response_text = self._extract_text_from_content(content)

            if not response_text:
                response_text = self._get_fallback_response(language)

            reminder_preview = self._build_pending_reminder_preview(
                user_id, session_id, state.get("language_code")
            )
            if reminder_preview:
                response_text = f"{response_text}\n\n{reminder_preview}"

            logger.info(f"   → Update response generated ({len(response_text)} chars)")

            return {
                "agent_response": response_text,
                "messages": result_messages,
            }

        except Exception as e:
            logger.error(f"Update agent execution failed: {e}")
            return {
                "agent_response": self._get_fallback_response(language),
                "error": str(e),
            }

    # ─────────────────────────────────────────────────────────────
    # NODE 4: VALIDATE OUTPUT (AI-driven)
    # ─────────────────────────────────────────────────────────────

    def _emit_suggestion_node(self, state: HybridState) -> Dict[str, Any]:
        """Deliver a ready smart_suggestions `message` verbatim (improve / evaluate).

        The tool already composed the full reply (mode-LLM opening + deterministic
        body); we output it unchanged instead of letting the main agent re-render,
        removing the free-render risk the prompt's VERBATIM rule used to guard.
        Reached only from ``_route_after_agent`` when this turn's smart_suggestions
        call returned a ready message; plan/followup/error/empty do not route here.
        """
        ss = state.get("smart_suggestions_data") or {}
        return {"response": (ss.get("message") or "").strip()}

    def _validate_output_node(self, state: HybridState) -> Dict[str, Any]:
        """Validate agent output using AI."""
        logger.info("✅ Step 5: Validating output...")

        response = state.get("agent_response", "")

        if isinstance(response, list):
            response = self._extract_text_from_content(response)

        if not isinstance(response, str):
            response = str(response) if response else ""

        if (
            response
            and response.strip().startswith("{")
            and response.strip().endswith("}")
        ):
            try:
                data = json.loads(response)
                if "response" in data:
                    response = data["response"]
            except json.JSONDecodeError:
                pass

        return {"response": response}

    # ─────────────────────────────────────────────────────────────
    # NODE 5: GENERATE RESPONSE
    # ─────────────────────────────────────────────────────────────

    async def _generate_response_node(self, state: HybridState) -> Dict[str, Any]:
        """Final response generation and orchestrating multi-action queue resumption."""
        await self._emit_progress(state, "finalizing")
        logger.info("📤 Step 5: Finalizing response...")

        quick_reply_result = state.get("quick_reply_result")

        if (
            quick_reply_result
            and quick_reply_result.get("status") == "multi_action_cancelled"
        ):
            return {
                "response": state.get("response", ""),
                "extracted_info": None,
                "quick_reply_buttons": None,
            }

        if quick_reply_result:
            response_data = await self._generate_quick_reply_response(
                state, quick_reply_result
            )
        else:
            response_data = {
                "response": state.get("response", ""),
                "extracted_info": self._extract_info_from_state(state),
            }
        user_id = state.get("user_id")
        session_id = state.get("session_id", "")
        if user_id and not state.get("multi_action_mode"):
            queues = await get_action_queues(user_id, session_id, self.memory_service)
            if queues and queues.get("action_queue"):
                action_queue = queues.get("action_queue", [])
                complete_action_queue = queues.get("complete_action_queue", [])
                original_message = queues.get("original_message", "")

                is_action_resolved = False

                if quick_reply_result:
                    status = quick_reply_result.get("status")
                    terminal_statuses = [
                        ToolStatus.SUCCESS,
                        ToolStatus.SYNC_SUCCESS,
                        ToolStatus.SYNC_SKIPPED,
                        ToolStatus.NO_SYNC_PENDING,
                        ToolStatus.CANCELLED,
                        ToolStatus.ERROR,
                    ]
                    sync_available = quick_reply_result.get("sync_available", False)
                    if status in terminal_statuses:
                        if status in [
                            ToolStatus.SYNC_SUCCESS,
                            ToolStatus.SYNC_SKIPPED,
                            ToolStatus.NO_SYNC_PENDING,
                        ]:
                            is_action_resolved = True
                        elif not sync_available:
                            is_action_resolved = True
                else:
                    pending_context = state.get("pending_action_context", {})
                    is_currently_pending = pending_context.get("has_pending", False)
                    is_newly_pending = state.get("has_pending_confirmation", False)

                    if not is_currently_pending and not is_newly_pending:
                        is_action_resolved = True

                if is_action_resolved and action_queue:
                    logger.info(
                        "🔄 Action fully resolved! Advancing Multi-Action Queue..."
                    )

                    completed_action = action_queue.pop(0)
                    if (
                        quick_reply_result
                        and quick_reply_result.get("status") == ToolStatus.ERROR
                    ):
                        completed_action["result"] = quick_reply_result.get(
                            "error", "Error via interactive flow."
                        )
                        completed_action["status"] = "error"
                    elif (
                        quick_reply_result
                        and quick_reply_result.get("status") == ToolStatus.CANCELLED
                    ):
                        completed_action["result"] = "Cancelled via interactive flow."
                        completed_action["status"] = "success"
                    else:
                        completed_action["result"] = "Completed via interactive flow."
                        completed_action["status"] = "success"
                    complete_action_queue.append(completed_action)

                    final_response = response_data.get("response", "")
                    language = state.get("language_name", "English")
                    language_code = state.get("language_code", "en")

                    if action_queue:
                        next_action = action_queue[0]
                        connective = await _generate_connective_tissue(
                            self.llm,
                            self._extract_text_from_content,
                            final_response,
                            next_action,
                            language,
                        )

                        if connective:
                            final_response = f"{connective}"
                        else:
                            next_tool = next_action.get("tool_name", "")
                            fallback_q = await _proceed_fallback(
                                self.llm,
                                self._extract_text_from_content,
                                next_tool,
                                language,
                            )
                            final_response = f"{fallback_q}"

                        response_data["quick_reply_buttons"] = (
                            self._create_quick_reply_buttons(
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
                        )
                        await save_action_queues(
                            user_id,
                            session_id,
                            action_queue,
                            complete_action_queue,
                            self.memory_service,
                            original_message,
                        )
                    else:
                        await delete_action_queues(
                            user_id, session_id, self.memory_service
                        )

                    response_data["response"] = final_response

        return response_data

    # ─────────────────────────────────────────────────────────────
    # HELPER METHODS
    # ─────────────────────────────────────────────────────────────

    def _build_agent_prompt(
        self,
        user_data: Dict[str, Any],
        timezone: str,
        language: str,
        real_value_context: RealValueContext,
        date_expressions: List[Dict[str, Any]],
        bot_name: str,
    ) -> str:
        """Build the agent system prompt."""

        # Current time
        current_time_str = format_current_time_prompt(timezone)

        merged_context = build_merged_user_context(
            user_data,
            user_data.get("ltm_retrieval_result"),
            base_profile_markdown=self._format_user_profile(user_data),
        )
        user_profile = merged_context.merged_profile_markdown
        write_summary = user_data.get("ltm_write_summary")
        if write_summary:
            user_profile = f"{user_profile}\n\n{write_summary}".strip()
        # Resolved date expressions section (kept separate from user profile)
        resolved_dates_section = self._format_resolved_date_expressions(
            date_expressions
        )

        # Append resolved dates right below current_time in the system prompt,
        # user_profile stays purely about the user.
        current_time_with_dates = (
            f"{current_time_str}\n{resolved_dates_section}\n\n"
            if resolved_dates_section
            else current_time_str
        )

        prompt = workflow_system_prompt(
            current_time=current_time_with_dates,
            language=language,
            user_profile=user_profile,
            bot_name=bot_name,
        )
        return prompt

    def _format_resolved_date_expressions(
        self, date_expressions: List[Dict[str, Any]]
    ) -> str:
        """Format resolved date expressions as a markdown section."""
        if not date_expressions:
            return ""

        lines = ["### Resolved Date Expressions"]
        for expr in date_expressions:
            raw = expr.get("raw_text") or expr.get("raw") or ""
            date_str = expr.get("date")
            start_date = expr.get("start_date")
            end_date = expr.get("end_date")
            dow = expr.get("day_of_week_resolved")
            start_dow = expr.get("start_day_of_week_resolved")
            end_dow = expr.get("end_day_of_week_resolved")

            if date_str:
                # Single resolved date, include day-of-week if available
                if dow:
                    lines.append(f'- "{raw}" → {date_str} ({dow})')
                else:
                    lines.append(f'- "{raw}" → {date_str}')
            elif start_date or end_date:
                # Date range, include day-of-week if available
                start_part = start_date or "?"
                end_part = end_date or "?"
                if start_dow:
                    start_part = f"{start_part} ({start_dow})"
                if end_dow:
                    end_part = f"{end_part} ({end_dow})"
                lines.append(f'- "{raw}" → {start_part} to {end_part}')

        return "\n".join(lines) if len(lines) > 1 else ""

    def _format_user_profile(self, user_data: Dict[str, Any]) -> str:
        """Format user profile for prompt - uses shared utility."""
        compact_user = compact_user_data(user_data)
        return format_user_data_as_markdown(compact_user)

    def _extract_info_from_state(self, state: HybridState) -> Optional[Dict[str, Any]]:
        """Extract additional info from state."""
        info = {}

        if state.get("user_data"):
            info["user_data"] = state["user_data"]
            info["flow"] = "general"

        # Look for tool results (e.g., set_goal)
        messages = state.get("messages", [])
        for msg in messages:
            if hasattr(msg, "name") and msg.name:
                try:
                    content = msg.content if hasattr(msg, "content") else str(msg)
                    if isinstance(content, str):
                        result = json.loads(content)
                        if (
                            isinstance(result, dict)
                            and result.get("success")
                            and result.get("goal")
                        ):
                            goal = result["goal"]
                            metric = goal.get("metric_name", "").lower()
                            value = goal.get("target_value")

                            goal_key = get_goal_key_from_metric(metric)
                            info["set_goal"] = {goal_key: value}
                except (json.JSONDecodeError, AttributeError):
                    pass

        return info if info else None

    def _extract_text_from_content(self, content) -> str:
        """
        Extract text from various content formats.
        """
        if content is None:
            return ""

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif "text" in item:
                        text_parts.append(item["text"])
                    elif "content" in item:
                        text_parts.append(str(item["content"]))
                elif isinstance(item, str):
                    text_parts.append(item)
            return " ".join(text_parts).strip()

        if isinstance(content, dict):
            return content.get("text", content.get("content", str(content)))

        return str(content)

    def _parse_json_response(self, content: str) -> dict:
        """Parse JSON from LLM response."""
        content = content.strip()

        # Remove markdown code blocks if present
        if "```" in content:
            parts = content.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    content = part
                    break

        # Find JSON object in content
        start_idx = content.find("{")
        end_idx = content.rfind("}") + 1
        if start_idx != -1 and end_idx > start_idx:
            content = content[start_idx:end_idx]

        return json.loads(content)

    def _get_fallback_response(self, language: str) -> str:
        """Get fallback response."""
        if language.lower() in ("vi", "vietnamese"):
            return "Mình có thể giúp gì cho bạn? 😊"
        return "How can I help you? 😊"

    def _get_error_response(self, language: str) -> str:
        """Get error response."""
        if language.lower() in ("vi", "vietnamese"):
            return "Xin lỗi, có lỗi xảy ra. Bạn thử lại nhé! 🙏"
        return "Sorry, something went wrong. Please try again! 🙏"

    def _is_single_word(self, text: str) -> bool:
        words = re.findall(r"\b\w+\b", text.strip())
        return len(words) == 1

    def _is_single_token(self, text: str) -> bool:
        return len(text.strip().split()) == 1

    def _should_keep_last_language(
        self,
        message: str,
        last_language_code: str,
        last_language: str,
        has_pending: bool,
    ) -> bool:
        """Decide whether to bypass language detection and keep previous language."""
        if is_iso_language_code(message) and last_language_code:
            logger.info(
                f"   → Message '{message}' looks like ISO language code, keeping last language: {last_language_code} ({last_language})"
            )
            return True

        if (
            has_pending
            and last_language
            and (self._is_single_word(message) or self._is_single_token(message))
        ):
            logger.info(
                f"   → Message is a single word, defaulting to Last Language Code {last_language_code} {last_language}"
            )
            return True

        return False

    def _detect_language_from_quick_reply(
        self, quick_reply: Dict[str, Any]
    ) -> Tuple[str, str]:
        """Detect language from quick_reply."""
        if quick_reply.get("language_code"):
            language_code = quick_reply.get("language_code")
            language_name = LANGUAGE_NAMES.get(language_code, "English")
            return language_code, language_name

        quick_reply_label = quick_reply.get("label", "")
        quick_reply_label_lower = quick_reply_label.lower()

        language_code = "en"
        language_name = "English"
        for lang_code, labels in QUICK_REPLY_BUTTON_LABELS.items():
            for label_key, label_value in labels.items():
                if quick_reply_label_lower == label_value.lower():
                    language_code = lang_code
                    language_name = LANGUAGE_NAMES.get(lang_code, "English")
                    break
            else:
                continue
            break

        return language_code, language_name

    def _get_last_ai_message(self, chat_history: List[Dict[str, Any]]) -> str:
        """Get the most recent AI message from chat history."""
        for msg in reversed(chat_history):
            if msg.get("role") == "assistant":
                return msg.get("content", "")
        return ""

    def _create_quick_reply_buttons(
        self, button_configs: List[Dict[str, str]], language_code: str
    ) -> List[Dict[str, Any]]:
        """Create quick_reply buttons with labels based on language."""
        labels = QUICK_REPLY_BUTTON_LABELS.get(
            language_code, QUICK_REPLY_BUTTON_LABELS["en"]
        )

        buttons = []
        for config in button_configs:
            value = config.get("value")
            label_key = config.get("label_key", value)

            button = {
                "value": value,
                "label": labels.get(label_key, value),
                "language_code": language_code,
            }

            icon = ICONS.get(value)
            if icon:
                button["icon"] = icon

            buttons.append(button)

        return buttons

    def _build_actions_summary(
        self, execution_results: List[Dict[str, Any]]
    ) -> Tuple[str, str]:
        """Build actions summary text from execution results."""
        actions_summary = []
        action_completed = ""
        if execution_results:
            for result in execution_results:
                tool_name = result.get("tool_name", "")
                exec_result = result.get(CommonConstant.RESULT, {})

                if tool_name == PendingActionTool.UPDATE_HEALTH_CONDITIONS:
                    actions_summary.append("updated health conditions")
                    action_completed = PendingActionTool.USER_PROFILE
                elif tool_name == PendingActionTool.UPDATE_DAILY_HABITS:
                    actions_summary.append("updated daily habits")
                    action_completed = PendingActionTool.USER_PROFILE
                elif tool_name == PendingActionTool.UPDATE_JOB_TITLE:
                    actions_summary.append("updated job title")
                    action_completed = PendingActionTool.USER_PROFILE
                elif tool_name == PendingActionTool.SET_HEALTH_GOAL:
                    actions_summary.append("set health goal")
                    action_completed = PendingActionTool.HEALTH_GOAL
                elif tool_name == PendingActionTool.SET_TIME_BASED_HEALTH_GOAL_SLEEP:
                    actions_summary.append("set time-based health goal sleep")
                    action_completed = PendingActionTool.HEALTH_GOAL
                elif (
                    tool_name
                    == PendingActionTool.SET_TIME_BASED_HEALTH_GOAL_ACTIVE_FOCUS
                ):
                    actions_summary.append("set time-based health goal active focus")
                    action_completed = PendingActionTool.HEALTH_GOAL
                elif tool_name == PendingActionTool.CREATE_EVENT_BY_NAME:
                    # Prefer created_count (events the backend confirmed by id) so the
                    # message reflects what was actually persisted, not what was asked.
                    requested = exec_result.get("requested_count")
                    created = exec_result.get("created_count")
                    if created is None:  # legacy callers without the new fields
                        created = exec_result.get("count", 0)
                    if requested is not None and created < requested:
                        actions_summary.append(
                            f"created {created} of {requested} event(s) "
                            f"(some could not be created)"
                        )
                    else:
                        actions_summary.append(f"created {created} event(s)")
                    action_completed = PendingActionTool.CALENDAR
                elif tool_name == PendingActionTool.UPDATE_EVENT_BY_NAME:
                    # Name the exact event updated so the reply can't borrow a name
                    # that only appeared as a CONFLICT in the prior overlap warning.
                    updated = (
                        exec_result.get("event", {})
                        if isinstance(exec_result, dict)
                        else {}
                    )
                    updated_name = (
                        updated.get("summary") if isinstance(updated, dict) else None
                    )
                    actions_summary.append(
                        f"updated the single event '{updated_name}'"
                        if updated_name
                        else "updated the single event"
                    )
                    action_completed = PendingActionTool.CALENDAR
                elif tool_name == PendingActionTool.DELETE_EVENT_BY_NAME:
                    count = exec_result.get("deleted_count") or exec_result.get(
                        "count", 0
                    )
                    actions_summary.append(f"deleted {count} event(s)")
                    action_completed = PendingActionTool.CALENDAR
                elif tool_name == PendingActionTool.ASK_ADD_MEETING_LINK:
                    actions_summary.append("added meeting link")
                    action_completed = PendingActionTool.CALENDAR
                elif tool_name == PendingActionTool.CREATE_FINANCE_LOGS:
                    count = exec_result.get("created_count", 0)
                    actions_summary.append(f"logged {count} finance entries")
                    action_completed = PendingActionTool.FINANCE
                elif tool_name == PendingActionTool.CREATE_REMINDER:
                    count = exec_result.get("count", 0)
                    actions_summary.append(f"created {count} reminder(s)")
                    action_completed = PendingActionTool.REMINDER
                elif tool_name == PendingActionTool.UPDATE_REMINDER:
                    actions_summary.append("updated reminder")
                    action_completed = PendingActionTool.REMINDER
                elif tool_name == PendingActionTool.DELETE_REMINDER:
                    count = exec_result.get("count", 0)
                    actions_summary.append(f"deleted {count} reminder(s)")
                    action_completed = PendingActionTool.REMINDER
                elif tool_name == PendingActionTool.MARK_OFF_REMINDER:
                    count = exec_result.get("count", 0)
                    actions_summary.append(f"marked off {count} reminder(s)")
                    action_completed = PendingActionTool.REMINDER
        return (
            ", ".join(actions_summary) if actions_summary else "completed the actions",
            action_completed,
        )

    def _build_success_status(
        self, quick_reply_result: Dict[str, Any], language_code: str
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Build status description and buttons for success status."""
        execution_results = quick_reply_result.get(CommonConstant.RESULT, [])
        actions_text, action_completed = self._build_actions_summary(execution_results)
        status_description = f" to confirm pending actions.\nThe following actions were successfully executed: {actions_text}"

        quick_reply_buttons = []
        sync_available = quick_reply_result.get("sync_available", False)
        sync_event_count = quick_reply_result.get("sync_event_count", 0)
        if sync_available and sync_event_count > 0:
            status_description += f"\n\nWould you like to sync {sync_event_count} event(s) to Google Calendar?"
            quick_reply_buttons = self._create_quick_reply_buttons(
                [
                    {"value": "sync_to_google", "label_key": "sync_to_google"},
                    {"value": "no_sync", "label_key": "no_sync"},
                ],
                language_code,
            )

        return status_description, quick_reply_buttons, action_completed

    def _build_cancelled_status(
        self, quick_reply_result: Dict[str, Any]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Build status description for cancelled status."""
        deleted_count = quick_reply_result.get("deleted_count", 0)
        status_description = f" to cancel pending actions.\n{deleted_count} pending action(s) were cancelled."
        return status_description, [], ""

    def _build_confirm_cancel_status(
        self, quick_reply_result: Dict[str, Any], language_code: str
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Build status description and buttons for confirm_cancel status."""
        pending_info = quick_reply_result.get("pending_info", {})
        summary = pending_info.get("summary", "")
        summaries_text = summary if summary else "pending action"
        status_description = f" to cancel.\nAre you sure you want to cancel {summaries_text}? You can also keep editing it."
        quick_reply_buttons = self._create_quick_reply_buttons(
            [
                {
                    "value": QuickReply.CONFIRM_CANCEL_PENDING,
                    "label_key": QuickReply.LABEL_KEY_CONFIRM_CANCEL,
                },
                {
                    "value": QuickReply.EDIT_PENDING,
                    "label_key": QuickReply.LABEL_KEY_EDIT,
                },
            ],
            language_code,
        )
        return status_description, quick_reply_buttons, ""

    def _build_editing_status(self) -> Tuple[str, List[Dict[str, Any]]]:
        """Build status description for editing status."""
        status_description = (
            " to edit the pending action.\nWhat changes would you like to make?"
        )
        return status_description, [], ""

    def _build_overlapping_status(
        self, quick_reply_result: Dict[str, Any], language_code: str
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Build status description and buttons for overlapping status."""
        overlapping_events = quick_reply_result.get("overlapping_events", [])
        pending_event = quick_reply_result.get("pending_event", {})
        overlap_summaries = [e.get("summary", "Event") for e in overlapping_events]
        overlap_text = (
            ", ".join(overlap_summaries) if overlap_summaries else "existing event(s)"
        )
        pending_summary = pending_event.get("summary", "Event")
        status_description = f" to confirm.\nThe event '{pending_summary}' overlaps with: {overlap_text}.\nWould you still like to proceed with booking?"
        quick_reply_buttons = self._create_quick_reply_buttons(
            [
                {
                    "value": "continue_with_overlapping",
                    "label_key": "continue_overlapping",
                },
                {"value": "cancel_overlapping", "label_key": "cancel_overlapping"},
            ],
            language_code,
        )
        return status_description, quick_reply_buttons, ""

    def _build_no_action_status(self) -> Tuple[str, List[Dict[str, Any]]]:
        """Build status description for no_action status."""
        status_description = ", but there were no pending actions to process."
        return status_description, [], ""

    def _build_partial_success_status(
        self, quick_reply_result: Dict[str, Any]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Build status description for partial_success status."""
        execution_results = quick_reply_result.get(CommonConstant.RESULT, [])
        errors = quick_reply_result.get(CommonConstant.ERROR, [])
        execution_count = len(execution_results) if execution_results else 0
        error_count = len(errors) if errors else 0
        status_description = f" to confirm pending actions.\n{execution_count} action(s) succeeded, but {error_count} action(s) failed."
        return status_description, [], ""

    def _build_sync_success_status(
        self, quick_reply_result: Dict[str, Any], language_code: str
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Build status description for sync_success status."""
        event_count = quick_reply_result.get("event_count", 0)
        meeting_link_results = quick_reply_result.get("result", [])
        ask_add_link = False
        meeting_link_message = ""
        if meeting_link_results:
            meeting_link_result = meeting_link_results[0].get("result", {})
            event_count = 1
            ask_add_link = meeting_link_result.get("ask_add_link", False)
            meeting_link_message = meeting_link_result.get("meeting_link_message", "")

        provider_name = quick_reply_result.get("provider_name", "Google")
        provider_display = provider_name.title() + " Calendar"
        status_description = f" to sync events to {provider_display}.\n{event_count} event(s) have been synced to {provider_display} successfully. {meeting_link_message}"

        if ask_add_link:
            quick_reply_buttons = self._create_quick_reply_buttons(
                [
                    {
                        "value": QuickReply.CONFIRM_ADD_MEETING_LINK,
                        "label_key": QuickReply.LABEL_KEY_ADD_MEETING_LINK,
                    },
                    {
                        "value": QuickReply.CANCEL_ADD_MEETING_LINK,
                        "label_key": QuickReply.LABEL_KEY_NO_MEETING_LINK,
                    },
                ],
                language_code,
            )
            return status_description, quick_reply_buttons, ""
        else:
            return status_description, [], ""

    def _build_sync_failed_status(
        self, quick_reply_result: Dict[str, Any]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Build status description for sync_failed status."""
        error_msg = quick_reply_result.get("error", "Unknown error")
        event_count = quick_reply_result.get("event_count", 0)
        provider_name = quick_reply_result.get("provider_name", "Google")
        provider_display = provider_name.title() + " Calendar"
        status_description = f" to sync events to {provider_display}.\nFailed to sync {event_count} event(s) to {provider_display}.\nError: {error_msg}"
        return status_description, [], ""

    def _build_sync_skipped_status(
        self, quick_reply_result: Dict[str, Any]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Build status description for sync_skipped status."""
        event_count = quick_reply_result.get("event_count", 0)
        status_description = f" to skip syncing to external calendar.\n{event_count} event(s) will not be synced."
        return status_description, [], ""

    def _build_no_sync_pending_status(self) -> Tuple[str, List[Dict[str, Any]]]:
        """Build status description for no_sync_pending status."""
        status_description = ", but there were no events pending for sync."
        return status_description, []

    def _build_error_status(
        self, quick_reply_result: Dict[str, Any]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Build status description for error status."""
        error_msg = quick_reply_result.get("error", "An error has occurred.")
        status_description = (
            f", but an error occurred while processing.\nError: {error_msg}"
        )
        return status_description, [], ""

    def _build_ask_send_email_status(
        self, quick_reply_result: Dict[str, Any], language_code: str
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Build status description for ask_send_email status."""
        participants = quick_reply_result.get("send_mail_participants", [])
        status_description = f" to delete event.\nWould you like to send notification to {participants} to inform them about the deletion?"
        quick_reply_buttons = self._create_quick_reply_buttons(
            [
                {
                    "value": QuickReply.LABEL_KEY_CANCEL_WITH_SEND_EMAIL,
                    "label_key": QuickReply.LABEL_KEY_CANCEL_WITH_SEND_EMAIL,
                },
                {
                    "value": QuickReply.LABEL_KEY_CANCEL_WITH_NO_SEND_EMAIL,
                    "label_key": QuickReply.LABEL_KEY_CANCEL_WITH_NO_SEND_EMAIL,
                },
            ],
            language_code,
        )
        return status_description, quick_reply_buttons, ""

    def _build_add_meeting_link_success_status(
        self,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Build status description for add_meeting_link_success status."""
        status_description = f"The meeting link has been added successfully."
        return status_description, [], ""

    def _build_no_add_meeting_link_status(
        self, quick_reply_result: Dict[str, Any]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Build status description for no_add_meeting_link status."""
        status_description = quick_reply_result.get("meeting_link_message", "")
        return status_description, [], ""

    def _build_status_description_and_buttons(
        self,
        status: str,
        quick_reply_result: Dict[str, Any],
        language_code: str,
    ) -> Tuple[str, List[Dict[str, Any]], str]:
        """
        Build status description and quick_reply buttons based on status.
        """
        status_handlers = {
            ToolStatus.NEED_CONFIRMATION: lambda: self._build_success_status(
                quick_reply_result, language_code
            ),
            ToolStatus.CANCELLED: lambda: self._build_cancelled_status(
                quick_reply_result
            ),
            ToolStatus.CONFIRM_CANCEL: lambda: self._build_confirm_cancel_status(
                quick_reply_result, language_code
            ),
            ToolStatus.EDITING: lambda: self._build_editing_status(),
            ToolStatus.OVERLAPPING: lambda: self._build_overlapping_status(
                quick_reply_result, language_code
            ),
            ToolStatus.NO_ACTION: lambda: self._build_no_action_status(),
            ToolStatus.PARTIAL_SUCCESS: lambda: self._build_partial_success_status(
                quick_reply_result
            ),
            ToolStatus.SYNC_SUCCESS: lambda: self._build_sync_success_status(
                quick_reply_result, language_code
            ),
            ToolStatus.SYNC_FAILED: lambda: self._build_sync_failed_status(
                quick_reply_result
            ),
            ToolStatus.SYNC_SKIPPED: lambda: self._build_sync_skipped_status(
                quick_reply_result
            ),
            ToolStatus.NO_SYNC_PENDING: lambda: self._build_no_sync_pending_status(),
            ToolStatus.SUCCESS: lambda: self._build_success_status(
                quick_reply_result, language_code
            ),
            ToolStatus.ASK_SEND_EMAIL: lambda: self._build_ask_send_email_status(
                quick_reply_result, language_code
            ),
            ToolStatus.ADD_MEETING_LINK_SUCCESS: lambda: self._build_add_meeting_link_success_status(),
            ToolStatus.NO_ADD_MEETING_LINK: lambda: self._build_no_add_meeting_link_status(
                quick_reply_result
            ),
        }

        handler = status_handlers.get(
            status, lambda: self._build_error_status(quick_reply_result)
        )
        return handler()

    async def _generate_quick_reply_response(
        self, state: HybridState, quick_reply_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate natural language response for quick_reply using LLM."""
        quick_reply = state.get("quick_reply", {})
        status = quick_reply_result.get(CommonConstant.STATUS, ToolStatus.ERROR)
        chat_history = state.get("chat_history", [])

        language_code, language_name = self._detect_language_from_quick_reply(
            quick_reply
        )

        last_ai_message = self._get_last_ai_message(chat_history)

        try:
            context_section = ""
            if last_ai_message:
                context_section = (
                    f"**Previous AI message (for context):**\n'{last_ai_message}'"
                )
            status_description, quick_reply_buttons, action_completed = (
                self._build_status_description_and_buttons(
                    status, quick_reply_result, language_code
                )
            )

            quick_reply_label = quick_reply.get("label", "")

            is_queue_pending = bool(
                state.get("multi_action_mode") and state.get("action_queue")
            )

            if not is_queue_pending and state.get("user_id"):
                queues = await get_action_queues(
                    state.get("user_id"),
                    state.get("session_id", ""),
                    self.memory_service,
                )
                if queues and queues.get("action_queue"):
                    is_queue_pending = True

            prompt = quick_reply_response_prompt(
                status=status,
                quick_reply_label=quick_reply_label,
                language_name=language_name,
                context_section=context_section,
                status_description=status_description,
                has_pending_actions=is_queue_pending,
                is_queue_pending=is_queue_pending,
            )

            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            response_text = self._extract_text_from_content(response.content)

            logger.info(
                f"   → Generated quick_reply response ({len(response_text)} chars)"
            )

            result = {
                HybridStateConstants.RESPONSE: response_text,
                HybridStateConstants.EXTRACTED_INFO: None,
                HybridStateConstants.LANGUAGE_CODE: language_code,
                HybridStateConstants.LANGUAGE_NAME: language_name,
            }

            if quick_reply_buttons:
                result[HybridStateConstants.QUICK_REPLY_BUTTONS] = quick_reply_buttons
            if action_completed:
                result[HybridStateConstants.ACTION_COMPLETED] = action_completed

            return result

        except Exception as e:
            logger.exception(f"Error generating quick_reply response: {e}")
            return {
                HybridStateConstants.RESPONSE: "I've processed your request.",
                HybridStateConstants.EXTRACTED_INFO: None,
                HybridStateConstants.LANGUAGE_CODE: language_code,
                HybridStateConstants.LANGUAGE_NAME: language_name,
                HybridStateConstants.ACTION_COMPLETED: "",
            }

    # ─────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────

    async def process(
        self,
        message: str,
        chat_history: List[Dict[str, str]] = None,
        user_id: str = None,
        session_id: str = None,
        app_id: str = None,
        user_data: Dict[str, Any] = None,
        timezone: Optional[str] = None,
        last_language_code: str = "",
        quick_reply: Optional[Dict[str, str]] = None,
        request_id: str = "",
    ) -> Tuple[str, Optional[Dict[str, Any]], str, str, Optional[List[Dict[str, str]]]]:
        """
        Process a message through the hybrid workflow.
        """
        chat_history = chat_history or []
        if not message and quick_reply:
            message = quick_reply.get("label", "")

        base_user_data = user_data or {}
        bot_name = BotConfig.get_bot_name(app_id)
        enriched_user_data = {**base_user_data}
        ctx_token = set_token_usage_context(
            TokenUsageContext(
                user_id=user_id or "",
                timezone=timezone or "UTC",
                language_code=last_language_code or "en",
                llm=self.llm,
            )
        )

        try:
            # Read = recalled context (prior turns) → awaited in merge_preprocess, overlapping
            # the language/intent/date fan-out. Write = this turn's facts → fire-and-forget from
            # merge_preprocess after ltm_retrieval_result is attached (known_memory plumbing).
            ltm_read_task = None
            if user_id and session_id and message:
                logger.info(
                    "LTM retrieval outcome=retrieval_called "
                    f"user_id={user_id} session_id={session_id}"
                )

                async def _read() -> Any:
                    return await self.memory_runtime.long_term_memory_service.retrieve_context(
                        user_id=user_id,
                        session_id=session_id,
                        query_text=message,
                    )

                ltm_read_task = asyncio.ensure_future(_read())

            initial_state = HybridState(
                user_message=message,
                chat_history=chat_history,
                user_id=user_id or "",
                session_id=session_id or "",
                request_id=request_id or "",
                app_id=app_id or "",
                bot_name=bot_name,
                user_data=enriched_user_data,
                timezone=timezone or "UTC",
                quick_reply=quick_reply,
                language_code="en",
                language_name="English",
                last_language_code=last_language_code,
                intent=None,
                real_values=None,
                messages=[],
                agent_response="",
                tool_results=[],
                quick_reply_result=None,
                pending_action_context=None,
                response="",
                extracted_info=None,
                quick_reply_buttons=None,
                error=None,
                date_expressions=None,
                action_queue=None,
                complete_action_queue=None,
                multi_action_mode=False,
                has_pending_confirmation=False,
                auto_confirm_sync_available=False,
                auto_confirm_sync_event_count=0,
                ltm_read_task=ltm_read_task,
                skip_ltm_write=bool(quick_reply),
                ltm_write_task=None,
            )

            logger.info(f"🚀 Starting hybrid workflow for user {user_id}")
            result = await self.graph.ainvoke(initial_state)
            logger.info("Base Hybrid workflow execution finalized.")

            return (
                result[HybridStateConstants.RESPONSE],
                result.get(HybridStateConstants.EXTRACTED_INFO),
                result.get(HybridStateConstants.LANGUAGE_CODE, "en"),
                result.get(HybridStateConstants.LANGUAGE_NAME, "English"),
                result.get(HybridStateConstants.QUICK_REPLY_BUTTONS),
                result.get(HybridStateConstants.ACTION_COMPLETED, ""),
            )

        except Exception as e:
            logger.exception(f"Hybrid workflow error: {e}")
            return self._get_error_response("English"), None, "en", "English", None, ""
        finally:
            reset_token_usage_context(ctx_token)


def is_iso_language_code(text: str) -> bool:
    """Check whether input text is a valid ISO alpha-2/alpha-3 language code."""
    if not isinstance(text, str):
        return False

    text = text.lower().strip()
    if not text.isalpha():
        return False

    if len(text) == 2:
        return pycountry.languages.get(alpha_2=text) is not None

    if len(text) == 3:
        return pycountry.languages.get(alpha_3=text) is not None

    return False

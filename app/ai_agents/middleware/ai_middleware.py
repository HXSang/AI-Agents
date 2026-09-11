"""
AI-Driven Middleware for Hybrid Workflow.

All validation and detection is done by AI - no hardcoded rules.
"""

import json
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union
from zoneinfo import ZoneInfo

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.ai_agents.constants import SmartSuggestionsConstants
from app.ai_agents.progress_labels import tool_step_label
from app.ai_agents.prompt import (
    workflow_output_validation_prompt,
    workflow_tool_validation_prompt,
)
from app.ai_agents.schemas.workflow_schemas import (
    LanguageDetectionResult,
    OutputValidationResult,
    RealValueContext,
    ToolValidationResult,
)
from app.config import settings
from app.utils.logger import logger


class ToolProgressMiddleware(AgentMiddleware):
    """Emit a coarse, localized progress label for each ReAct tool call.

    The main agent runs with only ``{"messages": ...}`` as its internal state, so the
    HybridWorkflow's request_id / user_id / session_id / language_code are NOT visible
    via ``request.state`` — they are injected through the constructor instead. Writes
    the shared response key tagged status="process" (overwritten by the final result).
    Best-effort: never raises into the tool call; unmapped tools leave the generic
    "processing" label.
    """

    def __init__(
        self,
        memory_service,
        user_id: str,
        session_id: str,
        request_id: str,
        language_code: str,
    ):
        self._mem = memory_service
        self._uid = user_id or ""
        self._sid = session_id or ""
        self._rid = request_id or ""
        self._lang = (language_code or "en").lower()

    async def _emit(self, request) -> None:
        try:
            if not (self._rid and self._uid):
                return
            try:
                tool_name = request.tool_call.get("name", "") or ""
            except Exception:
                tool_name = ""
            label = tool_step_label(tool_name, self._lang)
            if not label:
                return
            key = self._mem.get_response_key(self._uid, self._sid, self._rid)
            await self._mem.set_json(
                key,
                {
                    "status": "process",
                    "step": "tool",
                    "tool": tool_name,
                    "label": label,
                },
                ttl=settings.response_cache_ttl,
            )
        except Exception as e:
            logger.debug(f"tool progress emit skipped: {e}")

    async def awrap_tool_call(self, request, handler: Callable):
        await self._emit(request)
        return await handler(request)

    def wrap_tool_call(self, request, handler: Callable):
        # Sync path can't await the Redis write; just proceed. The workflow runs the
        # agent via ainvoke, so awrap_tool_call is the active path.
        return handler(request)


# ============================================================================
# HELPER FUNCTION: Extract text from various content formats
# ============================================================================


def extract_text_from_content(content: Union[str, list, Any]) -> str:
    """Extract plain text from various LLM response content formats.

    Handles:
    - Plain strings
    - Lists with reasoning_content and text blocks (Bedrock models)
    - Other formats (converts to string)

    Args:
        content: The content from LLM response (may be str, list, or other)

    Returns:
        Extracted plain text as string
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif item.get("type") == "reasoning_content":
                    # Skip reasoning content for final text extraction
                    pass
            elif isinstance(item, str):
                text_parts.append(item)
        return " ".join(text_parts).strip()

    return str(content)


# ============================================================================
# AI TOOL VALIDATION MIDDLEWARE
# ============================================================================


class AIToolValidationMiddleware(AgentMiddleware):
    """
    AI-driven tool parameter validation.
    Validates parameters using AI before tool execution.
    """

    # Tools that need validation (ID validation)
    # NOTE: create_event_by_name is NOT here because it has its own validator (CalendarCreateValidator)
    # that handles missing params and caching. This middleware validates IDs, not missing params.
    TOOLS_TO_VALIDATE = {}

    def __init__(self, llm, real_value_context: RealValueContext):
        """
        Args:
            llm: LLM for validation
            real_value_context: Context containing real values from API
        """
        self.llm = llm
        self.context = real_value_context

    def wrap_tool_call(self, request, handler: Callable):
        """Validate tool parameters before execution (sync version)."""
        tool_name = request.tool_call.get("name", "")
        args = request.tool_call.get("args", {})
        tool_call_id = request.tool_call.get("id", "")

        # For event/email/health read tools: wait for response and extract data immediately
        if tool_name in ("list_events", "get_today_events", "get_events"):
            # Execute tool and wait for response
            result = handler(request)
            # Extract events immediately after receiving response (before continuing)
            self._extract_events_from_result(result)
            logger.info(
                f"✅ Extracted {len(self.context.valid_event_ids)} event IDs from {tool_name} response"
            )
            self._track_tool_result(tool_name, result)
            return result
        elif tool_name == "get_reminders":
            result = handler(request)
            self._extract_reminders_from_result(result)
            logger.info(
                f"✅ Extracted {len(self.context.valid_reminder_ids)} reminder IDs from {tool_name} response"
            )
            self._track_tool_result(tool_name, result)
            return result
        elif tool_name == "get_emails":
            # Execute tool and wait for response
            result = handler(request)
            # Extract emails immediately after receiving response (before continuing)
            self._extract_emails_from_result(result)
            logger.info(
                f"✅ Extracted {len(self.context.valid_email_ids)} email IDs from {tool_name} response"
            )
            self._track_tool_result(tool_name, result)
            return result
        elif tool_name == "get_health_goal":
            # Execute tool and wait for response
            result = handler(request)
            # Extract health goals data immediately after receiving response (before continuing)
            self._extract_health_goals_from_result(result)
            logger.info(f"✅ Extracted health goals data from {tool_name} response")
            return result
        elif tool_name == "get_health_data":
            # Execute tool and wait for response
            result = handler(request)
            # Extract health data immediately after receiving response (before continuing)
            self._extract_health_data_from_result(result)
            logger.info(f"✅ Extracted health data from {tool_name} response")
            return result

        # Only validate specific tools
        if tool_name not in self.TOOLS_TO_VALIDATE:
            result = handler(request)
            self._track_tool_result(tool_name, result)
            return result

        # Validate with AI
        validation = self._validate_with_ai(tool_name, args)

        if validation and not validation.is_valid:
            logger.warning(f"⚠️ Tool validation failed: {validation.error_message}")

            return ToolMessage(
                content=json.dumps(
                    {
                        "error": validation.error_type or "validation_error",
                        "message": validation.error_message or "Validation failed",
                        "suggestion": validation.suggestion,
                        "detected_fake_values": validation.detected_fake_values or [],
                        "valid_event_ids": (
                            self.context.valid_event_ids[:5]
                            if self.context.valid_event_ids
                            else []
                        ),
                    }
                ),
                tool_call_id=tool_call_id,
            )

        # Valid - proceed with execution
        result = handler(request)
        self._track_tool_result(tool_name, result)
        return result

    async def awrap_tool_call(self, request, handler: Callable):
        """Validate tool parameters before execution (async version for ainvoke)."""
        tool_name = request.tool_call.get("name", "")
        args = request.tool_call.get("args", {})
        tool_call_id = request.tool_call.get("id", "")

        # For event/email/health read tools: wait for response and extract data immediately
        if tool_name in ("list_events", "get_today_events", "get_events"):
            # Execute tool and wait for response
            result = await handler(request)
            # Extract events immediately after receiving response (before continuing)
            self._extract_events_from_result(result)
            logger.info(
                f"✅ Extracted {len(self.context.valid_event_ids)} event IDs from {tool_name} response"
            )
            self._track_tool_result(tool_name, result)
            return result
        elif tool_name == "get_reminders":
            result = await handler(request)
            self._extract_reminders_from_result(result)
            logger.info(
                f"✅ Extracted {len(self.context.valid_reminder_ids)} reminder IDs from {tool_name} response"
            )
            self._track_tool_result(tool_name, result)
            return result
        elif tool_name == "get_emails":
            # Execute tool and wait for response
            result = await handler(request)
            # Extract emails immediately after receiving response (before continuing)
            self._extract_emails_from_result(result)
            logger.info(
                f"✅ Extracted {len(self.context.valid_email_ids)} email IDs from {tool_name} response"
            )
            self._track_tool_result(tool_name, result)
            return result
        elif tool_name == "get_health_goal":
            # Execute tool and wait for response
            result = await handler(request)
            # Extract health goals data immediately after receiving response (before continuing)
            self._extract_health_goals_from_result(result)
            logger.info(f"✅ Extracted health goals data from {tool_name} response")
            return result
        elif tool_name == "get_health_data":
            # Execute tool and wait for response
            result = await handler(request)
            # Extract health data immediately after receiving response (before continuing)
            self._extract_health_data_from_result(result)
            logger.info(f"✅ Extracted health data from {tool_name} response")
            return result

        # Only validate specific tools
        if tool_name not in self.TOOLS_TO_VALIDATE:
            result = await handler(request)
            self._track_tool_result(tool_name, result)
            return result

        # Validate with AI (reuse sync method - validation is fast)
        validation = self._validate_with_ai(tool_name, args)

        if validation and not validation.is_valid:
            logger.warning(f"⚠️ Tool validation failed: {validation.error_message}")

            return ToolMessage(
                content=json.dumps(
                    {
                        "error": validation.error_type or "validation_error",
                        "message": validation.error_message or "Validation failed",
                        "suggestion": validation.suggestion,
                        "detected_fake_values": validation.detected_fake_values or [],
                        "valid_event_ids": (
                            self.context.valid_event_ids[:5]
                            if self.context.valid_event_ids
                            else []
                        ),
                    }
                ),
                tool_call_id=tool_call_id,
            )

        # Valid - proceed with async execution
        result = await handler(request)
        self._track_tool_result(tool_name, result)
        return result

    def _validate_with_ai(self, tool_name: str, params: dict) -> ToolValidationResult:
        """Use AI to validate tool parameters."""
        try:
            # Build context string
            context_str = self._build_context_string()
            prompt = workflow_tool_validation_prompt()
            prompt = prompt.format(
                tool_name=tool_name,
                params=json.dumps(params, indent=2),
                real_values_context=context_str,
            )

            # LLM call with JSON parsing
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content_text = extract_text_from_content(response.content)
            data = self._parse_json(content_text)
            return ToolValidationResult(**data)

        except Exception as e:
            logger.error(f"AI tool validation failed: {e}")
            # Default to valid on error (don't block)
            return ToolValidationResult(is_valid=True, error_type="none")

    def _build_context_string(self) -> str:
        """Build context string for AI."""
        lines = []

        if self.context.valid_event_ids:
            lines.append(
                f"**Valid Event IDs (total: {len(self.context.valid_event_ids)}):**"
            )
            # Show all IDs for validation (not just first 10)
            for eid in self.context.valid_event_ids:
                name = self.context.event_id_to_name.get(eid, "Unknown")
                lines.append(f"  - '{eid}' → {name}")
        else:
            lines.append("**No events loaded yet (list_events not called)**")

        lines.append(f"\n**Tools already called:** {self.context.tools_called}")

        return "\n".join(lines)

    def _track_tool_result(self, tool_name: str, result):
        """Extract and track real values from tool results."""
        self.context.tools_called.append(tool_name)

    def _extract_events_from_result(self, result):
        """Extract real event IDs from list_events result."""
        try:
            content = result.content if hasattr(result, "content") else str(result)
            data = json.loads(content)

            events = data.get("events", [])
            new_event_ids = []

            for event in events:
                if isinstance(event, dict):
                    event_id = event.get("id") or event.get("event_id")
                    event_name = (
                        event.get("summary") or event.get("name") or event.get("title")
                    )

                    if event_id:
                        # Only add if not already in list (avoid duplicates)
                        if event_id not in self.context.valid_event_ids:
                            new_event_ids.append(event_id)
                            self.context.valid_event_ids.append(event_id)

                        # Always update name and details (in case event info changed)
                        if event_name:
                            self.context.event_id_to_name[event_id] = event_name
                        self.context.event_id_to_details[event_id] = event

        except Exception as e:
            logger.warning(f"Failed to extract events: {e}")

    def _extract_reminders_from_result(self, result):
        """Extract real reminder IDs + details from get_reminders result.

        Used to (a) prevent the agent from inventing reminder ids on
        update/delete/mark_off, and (b) backfill has_time/completed on a partial
        update / partition eligible mark-off ids.
        """
        try:
            content = result.content if hasattr(result, "content") else str(result)
            data = json.loads(content)

            reminders = data.get("reminders", [])
            for reminder in reminders:
                if isinstance(reminder, dict):
                    reminder_id = reminder.get("id") or reminder.get("reminder_id")
                    if reminder_id:
                        if reminder_id not in self.context.valid_reminder_ids:
                            self.context.valid_reminder_ids.append(reminder_id)
                        self.context.reminder_id_to_details[reminder_id] = reminder
        except Exception as e:
            logger.warning(f"Failed to extract reminders: {e}")

    def _extract_emails_from_result(self, result):
        """Extract real email IDs from get_emails result."""
        try:
            content = result.content if hasattr(result, "content") else str(result)
            data = json.loads(content)

            # API returns emails in "content" key (paginated response)
            emails = (
                data.get("content", [])
                or data.get("emails", [])
                or data.get("messages", [])
            )
            new_email_ids = []

            for email in emails:
                if isinstance(email, dict):
                    email_id = email.get("id") or email.get("message_id")
                    subject = email.get("subject")

                    if email_id:
                        # Only add if not already in list (avoid duplicates)
                        if email_id not in self.context.valid_email_ids:
                            new_email_ids.append(email_id)
                            self.context.valid_email_ids.append(email_id)

                        # Always update subject and details (in case email info changed)
                        if subject:
                            self.context.email_id_to_subject[email_id] = subject
                        # Store full details for validation
                        self.context.email_id_to_details[email_id] = email

            logger.info(
                f"✅ Extracted {len(new_email_ids)} new email IDs (total: {len(self.context.valid_email_ids)})"
            )

        except Exception as e:
            logger.warning(f"Failed to extract emails: {e}")

    def _extract_health_goals_from_result(self, result):
        """Extract health goals data from get_health_goal result."""
        try:
            content = result.content if hasattr(result, "content") else str(result)
            data = json.loads(content)

            # Response format: {"status": "authoritative_override", "data": formatted_goals}
            # formatted_goals is a markdown string
            health_goals_data = data.get("data", "")

            if health_goals_data:
                self.context.health_goals_data = health_goals_data
                logger.info(
                    f"✅ Extracted health goals data (length: {len(health_goals_data)} chars)"
                )
            else:
                logger.warning("No health goals data found in response")

        except Exception as e:
            logger.warning(f"Failed to extract health goals: {e}")

    def _extract_health_data_from_result(self, result):
        """Extract health data from get_health_data result."""
        try:
            content = result.content if hasattr(result, "content") else str(result)
            data = json.loads(content)

            # Response format: {"metric": "steps", "data": markdown_format}
            # markdown_format is a markdown string from _format_summary_as_markdown
            health_data = data.get("data", "")
            metric = data.get("metric", "all")

            if health_data:
                # Append to existing health_data if it exists, otherwise set it
                # This allows multiple metrics to be stored (e.g., steps, heart_rate, sleep, energy)
                if self.context.health_data:
                    self.context.health_data += "\n\n" + health_data
                else:
                    self.context.health_data = health_data
                logger.info(
                    f"✅ Extracted health data for metric '{metric}' (length: {len(health_data)} chars)"
                )
            else:
                logger.warning(
                    f"No health data found in response for metric '{metric}'"
                )

        except Exception as e:
            logger.warning(f"Failed to extract health data: {e}")

    def _parse_json(self, content: str) -> dict:
        """Parse JSON from LLM response."""
        content = content.strip()
        if "```" in content:
            parts = content.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    content = part
                    break
        return json.loads(content)


# ============================================================================
# AI OUTPUT VALIDATION MIDDLEWARE
# ============================================================================


class AIOutputValidationMiddleware(AgentMiddleware):
    """
    AI-driven output validation middleware.
    Validates agent response for hallucinations using AI.
    """

    def __init__(
        self,
        llm,
        real_value_context: RealValueContext,
        language: str,
        needs_real_ids: bool = True,
        timezone: str = "",
        bot_name: str = "Sylo",
    ):
        """
        Args:
            llm: LLM for validation
            real_value_context: Context containing real values
            language: User's language
            needs_real_ids: Whether to validate output (only validate if True)
        """
        self.llm = llm
        self.context = real_value_context
        self.language = language
        self.needs_real_ids = needs_real_ids
        self.timezone = timezone
        self.user_message = ""
        self.bot_name = bot_name

    def before_agent(self, state, runtime) -> Optional[Dict[str, Any]]:
        """Capture user message before agent runs."""
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "human":
                self.user_message = msg.content
                break
            if isinstance(msg, HumanMessage):
                self.user_message = msg.content
                break
        return None

    def after_model(
        self, state, response=None, runtime=None
    ) -> Optional[Dict[str, Any]]:
        """Validate model response after generation.

        Note: Signature is flexible to handle different LangChain versions.
        Response might be passed as second or third argument.
        Only validates if needs_real_ids is True.
        """
        # Skip validation if not needed
        if not self.needs_real_ids:
            return None

        # Skip validation if not tool_called
        if not self.context.tools_called:
            return None

        # Skip validation if create_event_by_name or get_health_goal was called (no need to check hallucinated IDs)
        if any(
            tool in self.context.tools_called
            for tool in [
                "create_event_by_name",
                "set_health_goal",
                "set_time_based_health_goal",
                "delete_event_by_name",
                "update_event_by_name",
                "sync_event_by_name",
                "get_reminders",
                "create_reminder",
                "update_reminder",
                "delete_reminder",
                "mark_off_reminder",
                "get_company_info",
                "smart_suggestions",
                "productivity_formulation_and_score",
                "balance_score_index_and_formula",
                "finance_summary_and_score",
                "log_finance_entries",
                "web_search",
            ]
        ):
            return None

        # Handle different calling conventions
        if response is None:
            # Try to get response from state
            messages = state.get("messages", [])
            if messages:
                response = messages[-1]
            else:
                return None

        # Skip tool calls - validate final response only
        if hasattr(response, "tool_calls") and response.tool_calls:
            return None

        if not hasattr(response, "content") or not response.content:
            return None

        # Validate with AI
        try:
            logger.info("Validating output with AI")
            # Extract only text content, skip reasoning_content
            text_content = extract_text_from_content(response.content)
            validation = self._validate_with_ai(text_content)

            if not validation or validation.is_valid:
                return None

            # Log issues
            if validation.has_hallucinations:
                logger.warning(
                    f"⚠️ Hallucinations detected: {validation.hallucinated_values}"
                )
            if validation.has_forbidden_content:
                logger.warning(f"⚠️ Forbidden content: {validation.forbidden_content}")

            # Regenerate if needed
            if validation.should_regenerate and validation.fixed_response:
                logger.info(
                    f"🔄 Using AI-fixed response for {validation.fixed_response}"
                )
                fixed_text = extract_text_from_content(validation.fixed_response)
                return {"messages": [AIMessage(content=fixed_text)]}
        except Exception as e:
            logger.warning(f"Output validation error: {e}")

        return None

    def _validate_with_ai(self, response: str) -> OutputValidationResult:
        """Use AI to validate response."""
        try:
            context_str = self._build_context_string()

            # Current time
            try:
                tz = ZoneInfo(self.timezone) if self.timezone else ZoneInfo("UTC")
                current_time = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")
            except Exception:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
            prompt = workflow_output_validation_prompt()
            prompt = prompt.format(
                current_time=current_time,
                timezone=self.timezone,
                response=response,
                user_message=self.user_message,
                real_values_context=context_str,
                language=self.language,
            )
            # LLM call with JSON parsing
            llm_response = self.llm.invoke([HumanMessage(content=prompt)])
            content_text = extract_text_from_content(llm_response.content)
            data = self._parse_json(content_text)
            return OutputValidationResult(**data)

        except Exception as e:
            logger.error(f"AI output validation failed: {e}")
            return OutputValidationResult(is_valid=True)

    def _build_context_string(self) -> str:
        """Build context string for AI with full details.

        This context is used in workflow_output_validation_prompt to validate
        that the agent's response only uses real event IDs from list_events/get_today_events/get_events.
        """
        lines = []

        if self.context.valid_event_ids:
            lines.append(
                f"**Real Events (ALL {len(self.context.valid_event_ids)} events are valid):**"
            )
            # Show all events (not just first 10) for comprehensive validation
            for idx, eid in enumerate(self.context.valid_event_ids, 1):
                details = self.context.event_id_to_details.get(eid, {})
                name = (
                    details.get("summary")
                    or details.get("name")
                    or details.get("title")
                    or "Unknown"
                )
                start = (
                    details.get("start_time")
                    or details.get("startTime")
                    or details.get("start", {}).get("dateTime", "")
                )
                end = (
                    details.get("end_time")
                    or details.get("endTime")
                    or details.get("end", {}).get("dateTime", "")
                )
                meeting_url = (
                    details.get("meetingUrl") or details.get("meeting_url") or ""
                )
                location = details.get("location") or ""
                description = details.get("description") or ""
                status = details.get("status") or ""
                participants = details.get("participants") or []
                is_synced_to_google = details.get("is_synced_to_google") or False
                is_synced_to_sylo = details.get("is_synced_to_sylo") or False

                lines.append(f"  Event #{idx}:")
                lines.append(f'    - Name: "{name}"')
                if start:
                    lines.append(f"    - Start: {start}")
                if end:
                    lines.append(f"    - End: {end}")
                if location:
                    lines.append(f"    - Location: {location}")
                if participants:
                    # Format participants list
                    participant_list = []
                    if isinstance(participants, list):
                        for p in participants:
                            if isinstance(p, dict):
                                name = p.get("displayName") or p.get("name") or ""
                                email = p.get("email") or ""
                                if name and email:
                                    participant_list.append(f"{name} ({email})")
                                elif email:
                                    participant_list.append(email)
                                elif name:
                                    participant_list.append(name)
                            elif isinstance(p, str):
                                participant_list.append(p)
                    if participant_list:
                        participants_str = ", ".join(participant_list)
                        lines.append(f"    - Participants: {participants_str}")
                if description:
                    lines.append(f"    - Description: {description[:100]}")
                if status:
                    lines.append(f"    - Status: {status}")
                if is_synced_to_sylo:
                    lines.append(
                        f"    - Synced to {self.bot_name}: {is_synced_to_sylo}"
                    )
                if meeting_url:
                    lines.append(f"    - Meeting link: {meeting_url}")
                else:
                    lines.append(f"    - Synced to Google: {is_synced_to_google}")
                lines.append(f"    - ID: '{eid}'")
        else:
            lines.append(
                "**No events loaded (list_events, get_today_events, or get_events not called yet)**"
            )

        if self.context.valid_email_ids:
            lines.append(
                f"\n**Real Emails (ALL {len(self.context.valid_email_ids)} emails are valid):**"
            )
            # Show all emails (not just first 5) for comprehensive validation
            for idx, eid in enumerate(self.context.valid_email_ids, 1):
                details = self.context.email_id_to_details.get(eid, {})
                subject = details.get("subject", "Unknown")
                sender_name = (
                    details.get("senderName") or details.get("sender_name") or "Unknown"
                )
                sender_email = (
                    details.get("senderEmail") or details.get("sender_email") or ""
                )
                email_date = (
                    details.get("emailDate")
                    or details.get("email_date")
                    or details.get("date")
                    or ""
                )
                is_read = details.get("read") or details.get("is_read") or False

                lines.append(f"  Email #{idx}:")
                lines.append(f'    - Subject: "{subject}"')
                if sender_name or sender_email:
                    lines.append(f"    - Sender: {sender_name} ({sender_email})")
                if email_date:
                    lines.append(f"    - Date: {email_date}")
                lines.append(f"    - Read: {is_read}")
                lines.append(f"    - ID: '{eid}'")
        else:
            lines.append("\n**No emails loaded (get_emails not called yet)**")

        if self.context.health_goals_data:
            lines.append("\n**Health Goals Data (from get_health_goal):**")
            lines.append(self.context.health_goals_data)
        else:
            lines.append(
                "\n**No health goals loaded (get_health_goal not called yet)**"
            )

        if self.context.health_data:
            lines.append("\n**Health Data (from get_health_data):**")
            lines.append(self.context.health_data)
        else:
            lines.append("\n**No health data loaded (get_health_data not called yet)**")

        return "\n".join(lines)

    def _parse_json(self, content: str) -> dict:
        """Parse JSON from LLM response."""
        content = content.strip()
        if "```" in content:
            parts = content.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    content = part
                    break
        return json.loads(content)


# ============================================================================
# HELPER FUNCTIONS FOR CONTENT FILTERING MIDDLEWARE
# ============================================================================


def _get_response_from_state(state, response) -> Optional[Any]:
    """Get response from state if not provided directly.

    Args:
        state: Agent state
        response: Response object (may be None)

    Returns:
        Response object or None if not found
    """
    if response is None:
        messages = state.get("messages", [])
        if messages:
            return messages[-1]
        return None
    return response


def _copy_metadata_to_message(
    source,
    target: AIMessage,
    skip_fields: Optional[List[str]] = None,
) -> None:
    """Copy metadata attributes from source to target AIMessage.

    Args:
        source: Source response object
        target: Target AIMessage to copy metadata to
        skip_fields: Optional list of field names to skip (e.g., ["tool_calls"])
    """
    skip_fields = skip_fields or []

    # Define all metadata fields to copy
    metadata_fields = [
        "tool_calls",
        "id",
        "additional_kwargs",
        "response_metadata",
        "usage_metadata",
    ]

    for field in metadata_fields:
        # Skip if field is in skip list
        if field in skip_fields:
            continue

        # Check if source has the attribute
        if not hasattr(source, field):
            continue
        # Copy the value
        source_value = getattr(source, field)
        setattr(target, field, source_value)


def _update_state_messages(state, new_response: AIMessage) -> Dict[str, Any]:
    """Update messages in state with new response.

    Args:
        state: Agent state
        new_response: New AIMessage to replace last message

    Returns:
        Updated state dict with new messages
    """
    messages = list(state.get("messages", []))
    if messages:
        messages[-1] = new_response
        return {"messages": messages}
    return {}


# ============================================================================
# REASONING CONTENT FILTER MIDDLEWARE
# ============================================================================


class ReasoningContentFilterMiddleware(AgentMiddleware):
    """
    Filters out reasoning_content from model responses.
    Only removes reasoning_content blocks, keeps all other content types.
    """

    def after_model(
        self, state, response=None, runtime=None
    ) -> Optional[Dict[str, Any]]:
        """Filter reasoning_content from model response content."""
        # Get response from state if not provided
        response = _get_response_from_state(state, response)
        if response is None:
            return None

        if not hasattr(response, "content") or not response.content:
            return None

        # Filter reasoning_content from list content
        content = response.content
        if isinstance(content, list):
            filtered_content = []
            for item in content:
                # Only skip reasoning_content, keep everything else
                if isinstance(item, dict) and item.get("type") == "reasoning_content":
                    continue
                else:
                    # Keep all other types (text, tool_use, image, etc.)
                    filtered_content.append(item)

            # Update response content only if we filtered something
            if len(filtered_content) != len(content):
                new_response = AIMessage(content=filtered_content)
                _copy_metadata_to_message(response, new_response)
                return _update_state_messages(state, new_response)

        return None


class ToolChoicesValidationMiddleware(AgentMiddleware):
    """
    Deduplicate tool calls and enforce a single argument set per tool.

    - If the model emits the same tool multiple times with IDENTICAL args:
      keep only the first occurrence.
    - If the same tool appears with DIFFERENT args in the same response:
      keep only the first args and drop all later calls with different args.
    - This middleware only runs on the FIRST after_model pass; subsequent
      passes are ignored.
    """

    def __init__(self, tools: Optional[List[Any]] = None):
        """
        Args:
            tools: Optional list of tool definitions for validation
        """
        # Ensure we only process the first after_model call
        self._has_run = False
        self.tools = tools or []
        # Build tool name to expected args mapping (schema-based)
        self._tool_args_map = self._build_tool_args_map()

    def _build_tool_args_map(self) -> Dict[str, set]:
        """Build mapping of tool names to expected argument names.

        Extracts tool definitions from tools passed to middleware (from hybrid_workflow.py).
        LangChain @tool decorator creates a Pydantic model from function signature,
        which is accessible via tool.args_schema.model_fields.
        """
        tool_args_map = {}

        # If tools are provided, extract from tool definitions
        for tool in self.tools:
            if not hasattr(tool, "name"):
                continue

            tool_name = tool.name
            expected_args: set[str] = set()

            # LangChain @tool decorator creates a Pydantic model from function signature
            # The args_schema is a Pydantic model class with model_fields attribute
            if hasattr(tool, "args_schema") and tool.args_schema:
                # Method 1: Extract from Pydantic model_fields (primary method for @tool decorator)
                if hasattr(tool.args_schema, "model_fields"):
                    # Pydantic v2: model_fields is a dict mapping field names to FieldInfo
                    expected_args = set(tool.args_schema.model_fields.keys())
                # Method 2: Fallback to __annotations__ for TypedDict or other types
                elif hasattr(tool.args_schema, "__annotations__"):
                    expected_args = set(tool.args_schema.__annotations__.keys())

            # Always register the tool, even if it has no args (0-arg tools)
            tool_args_map[tool_name] = expected_args
            if expected_args:
                logger.debug(
                    f"📦 Extracted tool '{tool_name}' with {len(expected_args)} args: {sorted(expected_args)}"
                )
            else:
                logger.debug(
                    f"📦 Extracted tool '{tool_name}' with 0 args (no-arg tool)"
                )

        logger.info(
            f"✅ Built tool args map: {len(tool_args_map)}/{len(self.tools)} tools extracted"
        )

        return tool_args_map

    def _validate_tool_args(self, tool_name: str, args: Dict[str, Any]) -> bool:
        """Validate args against tool schema (unexpected fields only).

        Returns:
            True if args are acceptable, False otherwise.
        """
        # Get expected args for this tool
        expected_args = self._tool_args_map.get(tool_name)

        # If we don't know this tool, allow it (might be a new tool)
        if expected_args is None:
            logger.warning(
                f"⚠️ Unknown tool '{tool_name}' - allowing tool call (no validation)"
            )
            return True

        provided_args = set(args.keys())
        unexpected_args = provided_args - expected_args

        if unexpected_args:
            logger.warning(
                f"❌ Invalid tool call '{tool_name}': unexpected arguments {unexpected_args}. "
                f"Expected: {expected_args}, Got: {provided_args}"
            )
            return False

        # Allow partial args (required vs optional handled by tool schema when executing)
        return True

    def after_model(
        self, state, response=None, runtime=None
    ) -> Optional[Dict[str, Any]]:
        # Only process once
        if self._has_run:
            return None

        response = _get_response_from_state(state, response)
        if response is None:
            return None

        # Only process if there are tool_calls
        if not hasattr(response, "tool_calls") or not response.tool_calls:
            return None

        original_tool_calls = list(response.tool_calls)
        if not original_tool_calls:
            return None

        logger.info(f"🔍 Original tool calls: {original_tool_calls}")

        def _extract_name_and_args(tool_call: Any) -> tuple[str, Dict[str, Any]]:
            # LangChain ToolCall objects usually have .name and .args
            name = getattr(tool_call, "name", None)
            args = getattr(tool_call, "args", None)

            # Fallback for dict-style tool calls
            if name is None and isinstance(tool_call, dict):
                name = tool_call.get("name")
                args = tool_call.get("args")

            if args is None:
                args = {}

            # If args is JSON string, try to parse (best-effort)
            if isinstance(args, str):
                try:
                    import json as _json

                    parsed = _json.loads(args)
                    if isinstance(parsed, dict):
                        args = parsed
                except Exception:
                    # Leave as-is if parsing fails
                    pass

            if not isinstance(args, dict):
                # Ensure args is at least a dict for comparison
                args = {"_raw": args}

            return name or "", args

        canonical_args_by_tool: Dict[str, Dict[str, Any]] = {}
        kept_tools: list[Any] = []

        for tool_call in original_tool_calls:
            tool_name, args = _extract_name_and_args(tool_call)

            if not tool_name:
                # Skip malformed tool calls without name
                continue

            # Step 1: validate args against tool schema (unexpected keys)
            if not self._validate_tool_args(tool_name, args):
                # Invalid according to schema → drop
                continue

            # Step 2: deduplicate and enforce single canonical args per tool
            if tool_name not in canonical_args_by_tool:
                # First time we see this tool: set canonical args and keep this call
                canonical_args_by_tool[tool_name] = args
                kept_tools.append(tool_call)
                continue

            canonical_args = canonical_args_by_tool[tool_name]

            if args == canonical_args:
                # Duplicate with same args → drop to de-duplicate
                logger.info(
                    f"🧹 Dropping duplicate tool call '{tool_name}' with identical args"
                )
                continue

            # Different args for the same tool name → drop
            logger.info(
                f"🧹 Dropping tool call '{tool_name}' with different args than canonical"
            )

        # If nothing changed, do not touch the state
        if len(kept_tools) == len(original_tool_calls):
            self._has_run = True
            return None

        # Build new AIMessage with filtered tool_calls, keep content as-is
        new_response = AIMessage(content=response.content)
        # Copy metadata but skip tool_calls so we can set our filtered list
        _copy_metadata_to_message(response, new_response, skip_fields=["tool_calls"])
        new_response.tool_calls = kept_tools

        self._has_run = True
        logger.info(f"🔍 Kept tools: {kept_tools}")
        return _update_state_messages(state, new_response)


class ToolBoundaryMiddleware(AgentMiddleware):
    """
    Ensures clean tool call boundaries.
    If tool_calls exist, only keep tool_calls in AIMessage.
    Removes all content blocks (reasoning_content, text, tool_use) from content.
    """

    def after_model(
        self, state, response=None, runtime=None
    ) -> Optional[Dict[str, Any]]:
        """Clean tool call boundaries - keep only tool_calls if they exist.
        Also validates and filters invalid tool calls.
        """
        # Get response from state if not provided
        response = _get_response_from_state(state, response)
        if response is None:
            return None

        # Only process if there are tool_calls
        if not hasattr(response, "tool_calls") or not response.tool_calls:
            return None

        # Check if content needs cleaning
        needs_cleaning = False
        content = response.content

        # If content is a list, check for reasoning_content, text, or tool_use blocks
        if isinstance(content, list):
            has_unwanted_blocks = any(
                isinstance(item, dict)
                and item.get("type") in ("reasoning_content", "text", "tool_use")
                for item in content
            )
            if has_unwanted_blocks:
                needs_cleaning = True
        elif content and isinstance(content, str):
            # If content is non-empty string and we have tool_calls, clear it
            needs_cleaning = True

        # Only update if cleaning is needed
        if needs_cleaning:
            # Create new AIMessage with only tool_calls, no content
            new_response = AIMessage(content="")
            # Copy metadata but skip tool_calls since it's already set
            _copy_metadata_to_message(response, new_response)
            return _update_state_messages(state, new_response)

        return None


# ============================================================================
# SMART SUGGESTIONS HAND-OFF MIDDLEWARE
# ============================================================================


class SuggestionHandoffMiddleware(AgentMiddleware):
    """
    Captures the structured result of the ``smart_suggestions`` tool and — for
    plan-mode suggestions — stops the main agent from self-executing the
    suggested actions, so the workflow can hand them off to the multi-action
    pipeline instead.

    Two responsibilities:
    1. ``(a)wrap_tool_call``: when ``smart_suggestions`` runs, parse its JSON
       result and stash it on ``context.smart_suggestions_data``.
    2. ``after_model``: once a *plan* suggestion has been captured, strip any
       subsequent write tool_calls so the ReAct loop ends without writing —
       ``_route_after_agent`` then routes to ``extract_multi_actions``.

    Improve/evaluate modes carry no ``action_params``; they are captured but
    NOT blocked, so the agent answers with text as before.
    """

    TOOL_NAME = SmartSuggestionsConstants.TOOL_NAME

    # Mutation tools that must never run uninspected alongside smart_suggestions.
    # If the model emits one of these in the SAME AIMessage batch as
    # smart_suggestions (parallel tool calls), it would write directly and
    # bypass the multi-action confirm gate — and then the plan hand-off would
    # queue the *same* actions again, causing duplicates. We defer them here.
    WRITE_TOOLS = frozenset(
        {
            "create_event_by_name",
            "update_event_by_name",
            "delete_event_by_name",
            "sync_event_by_name",
            "create_reminder",
            "update_reminder",
            "delete_reminder",
            "mark_off_reminder",
            "set_health_goal",
            "set_time_based_health_goal",
            "create_finance_logs",
            "log_finance_entries",
        }
    )

    def __init__(self, real_value_context: RealValueContext):
        self.context = real_value_context

    # ── capture (sync + async) ───────────────────────────────────────────
    def wrap_tool_call(self, request, handler: Callable):
        if self._defer_sibling_write(request):
            return self._deferred_message(request)
        result = handler(request)
        if request.tool_call.get("name", "") == self.TOOL_NAME:
            self._capture(result)
        return result

    async def awrap_tool_call(self, request, handler: Callable):
        if self._defer_sibling_write(request):
            return self._deferred_message(request)
        result = await handler(request)
        if request.tool_call.get("name", "") == self.TOOL_NAME:
            self._capture(result)
        return result

    # ── same-batch guard (parallel tool-call leak) ───────────────────────
    def _defer_sibling_write(self, request) -> bool:
        """True if the current call is a write tool emitted in the SAME
        AIMessage batch as smart_suggestions.

        Decided from the static tool_calls list of the triggering AIMessage —
        so it holds regardless of the order parallel tools happen to execute,
        and regardless of whether smart_suggestions has been captured yet.
        """
        tool_name = request.tool_call.get("name", "")
        if tool_name not in self.WRITE_TOOLS:
            return False
        siblings = self._sibling_tool_names(request)
        if self.TOOL_NAME in siblings:
            logger.info(
                f"[SS-HANDOFF] step 2b | block sibling write {tool_name!r}: "
                f"emitted in same batch as {self.TOOL_NAME} -> defer to plan"
            )
            return True
        return False

    def _sibling_tool_names(self, request) -> set:
        """Names of tool_calls in the SAME AIMessage as the current call."""
        try:
            state = request.state
            messages = (
                state.get("messages", [])
                if isinstance(state, dict)
                else getattr(state, "messages", []) or []
            )
            current_id = request.tool_call.get("id")
            for msg in reversed(messages):
                tool_calls = getattr(msg, "tool_calls", None)
                if not tool_calls:
                    continue
                ids = {tc.get("id") for tc in tool_calls}
                if current_id in ids:
                    return {tc.get("name") for tc in tool_calls}
        except Exception as e:
            logger.warning(f"[SS-HANDOFF] step 2b · sibling scan failed: {e}")
        return set()

    def _deferred_message(self, request) -> ToolMessage:
        return ToolMessage(
            content=json.dumps(
                {
                    "status": "deferred",
                    "message": (
                        "Not executed: this action is being handled through the "
                        "suggestion plan (user will confirm). Do not retry now."
                    ),
                }
            ),
            tool_call_id=request.tool_call.get("id", ""),
        )

    def _capture(self, result) -> None:
        try:
            content = result.content if hasattr(result, "content") else result
            if isinstance(content, list):
                content = extract_text_from_content(content)
            if not isinstance(content, str):
                content = str(content)
            data = json.loads(content)
            if isinstance(data, dict):
                self.context.smart_suggestions_data = data
                mode = data.get("mode")
                n = len(data.get("suggestions") or [])
                tail = (
                    "-> will hand off to multi-action"
                    if mode == "plan" and n
                    else "-> no hand-off (agent answers as text)"
                )
                logger.info(
                    f"[SS-HANDOFF] step 1 | capture tool result: "
                    f"mode={mode!r} suggestions={n} {tail}"
                )
        except Exception as e:
            logger.warning(
                f"[SS-HANDOFF] step 1 · capture FAILED to parse tool result: {e}"
            )

    # ── block self-execution for plan mode ───────────────────────────────
    def after_model(
        self, state, response=None, runtime=None
    ) -> Optional[Dict[str, Any]]:
        ss = self.context.smart_suggestions_data
        # Block once a plan is IN HAND — a day/week plan with `suggestions`, OR a
        # read-only WEEK message with `groups` (which must book nothing). The no-plan
        # envelopes (too_late_today / day_part_passed / scope_unsupported / error) also
        # say "plan" but carry neither, so they fall through to the agent.
        if not ss or ss.get("mode") != "plan":
            return None
        if not (ss.get("suggestions") or ss.get("groups")):
            return None

        response = _get_response_from_state(state, response)
        if response is None:
            return None

        tool_calls = getattr(response, "tool_calls", None)
        if not tool_calls:
            return None

        # A plan suggestion is already in hand → stop the agent from writing.
        blocked = [tc.get("name") for tc in tool_calls]
        logger.info(
            f"[SS-HANDOFF] step 2 | block agent self-write: "
            f"dropped tool_calls={blocked}"
        )
        new_response = AIMessage(content="")
        _copy_metadata_to_message(response, new_response, skip_fields=["tool_calls"])
        # Defensive: ensure no provider-raw tool_calls linger in kwargs.
        if isinstance(getattr(new_response, "additional_kwargs", None), dict):
            new_response.additional_kwargs.pop("tool_calls", None)
        return _update_state_messages(state, new_response)


# ============================================================================
# REAL VALUE CONTEXT INJECTION MIDDLEWARE
# ============================================================================


class RealValueContextMiddleware(AgentMiddleware):
    """
    Injects real values into model context.
    Helps model use correct IDs instead of making them up.
    """

    def __init__(self, real_value_context: RealValueContext):
        """
        Args:
            real_value_context: Context containing real values
        """
        self.context = real_value_context

    def before_model(self, state, runtime) -> Optional[Dict[str, Any]]:
        """Inject real values before model call."""
        if (
            not self.context.valid_event_ids
            and not self.context.valid_email_ids
            and not self.context.valid_reminder_ids
        ):
            return None

        # Build context message
        context_msg = self._build_context_message()

        # Get messages and inject context
        messages = list(state.get("messages", []))

        if not messages:
            return None

        # Find position to insert (before last human message)
        insert_pos = len(messages) - 1
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], HumanMessage):
                insert_pos = i
                break

        messages.insert(insert_pos, SystemMessage(content=context_msg))

        return {"messages": messages}

    def _build_context_message(self) -> str:
        """Build context message with real values."""
        lines = ["**REAL VALUES FROM API (use these ONLY):**"]

        if self.context.valid_event_ids:
            lines.append("\n**Available Events:**")
            for eid in self.context.valid_event_ids:
                name = self.context.event_id_to_name.get(eid, "Unknown")
                lines.append(f"  - {name}: event_id='{eid}'")
            lines.append("\n⚠️ For delete/update: Use ONLY these event_id values")

        if self.context.valid_email_ids:
            lines.append("\n**Available Emails:**")
            for eid in self.context.valid_email_ids[:5]:
                subject = self.context.email_id_to_subject.get(eid, "Unknown")
                lines.append(f"  - {subject}")

        if self.context.valid_reminder_ids:
            lines.append("\n**Available Reminders:**")
            for rid in self.context.valid_reminder_ids:
                details = self.context.reminder_id_to_details.get(rid, {})
                title = (
                    details.get("title", "Unknown")
                    if isinstance(details, dict)
                    else "Unknown"
                )
                lines.append(f"  - {title}: reminder_id='{rid}'")
            lines.append(
                "\n⚠️ For delete/update/mark_off: Use ONLY these reminder_id values"
            )

        return "\n".join(lines)

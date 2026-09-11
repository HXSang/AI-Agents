import hashlib
import time
import uuid
from typing import Any, Dict, Optional

from fastapi import HTTPException

from models.models import ChatMessage
from services.chat_service import ChatService

from . import log_error, log_exception, log_info, log_warn
from .correlation import CorrelationManager
from .errors import AgentEmptyResponseError, InsightTimeoutError, PublishError
from .prompts import (
    HEALTH_INSIGHT_PROMPT_TEMPLATE,
    OVERALL_INSIGHT_PROMPT_TEMPLATE,
    PRODUCTIVITY_INSIGHT_PROMPT_TEMPLATE,
    max_chars,
    target_language
)


PROSE_MIN_REASONABLE_LEN = 20
PROSE_MAX_REASONABLE_LEN = 1500
PREVIEW_CHARS = 80

_HEALTH_SESSION_PREFIX: str = "chatbot-cdi-health-"
_PRODUCTIVITY_SESSION_PREFIX: str = "chatbot-cdi-productivity-"
_OVERALL_SESSION_PREFIX: str = "chatbot-cdi-overall-"


def _stable_session_id(prefix: str, user_id: str) -> str:
    digest = hashlib.sha256(f"{prefix}{user_id}".encode("utf-8")).hexdigest()
    return f"{digest[0:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"


class ChatDrivenHealthInsightService:

    APP_ID: str = "chat_driven_insights.health"
    APP_ID_PRODUCTIVITY: str = "chat_driven_insights.productivity"
    APP_ID_OVERALL: str = "chat_driven_insights.overall"
    FLOW: str = "general"

    def __init__(
        self,
        *,
        chat_service: ChatService,
        redis_client: Any,
        correlation: Optional[CorrelationManager] = None,
        timeout_sec: int = 120,
        poll_interval_sec: float = 0.3,
    ) -> None:
        self.chat_service = chat_service
        self.redis_client = redis_client
        self.correlation = correlation or CorrelationManager(
            redis_client=redis_client,
            timeout_sec=timeout_sec,
            poll_interval_sec=poll_interval_sec,
        )
        log_info(
            "INI",
            "ChatDrivenHealthInsightService init",
            chat_service=type(chat_service).__name__,
            redis_client=type(redis_client).__name__,
            correlation=type(self.correlation).__name__,
            timeout_sec=timeout_sec,
            poll_interval_sec=poll_interval_sec,
        )

    def set_external_api_service(self, _external_api_service: Any) -> None:
        return

    async def analyze_health_insight(
        self,
        *,
        user_id: str,
        timezone: Optional[str] = None,
        provider_name: Optional[str] = "",
        language: Optional[str] = "en-US",
        force_update: bool = False,
    ) -> Dict[str, Any]:
        """Return ``{"status","user_id","insight","error"}`` for the router."""
        return await self._analyze(
            prompt_template=HEALTH_INSIGHT_PROMPT_TEMPLATE,
            session_prefix=_HEALTH_SESSION_PREFIX,
            stage_prefix="health",
            app_id=self.APP_ID,
            user_id=user_id,
            timezone=timezone,
            provider_name=provider_name,
            language=language,
            force_update=force_update,
        )

    async def analyze_productivity_insight(
        self,
        *,
        user_id: str,
        timezone: Optional[str] = None,
        provider_name: Optional[str] = "",
        language: Optional[str] = "en-US",
        force_update: bool = False,
    ) -> Dict[str, Any]:
        return await self._analyze(
            prompt_template=PRODUCTIVITY_INSIGHT_PROMPT_TEMPLATE,
            session_prefix=_PRODUCTIVITY_SESSION_PREFIX,
            stage_prefix="productivity",
            app_id=self.APP_ID_PRODUCTIVITY,
            user_id=user_id,
            timezone=timezone,
            provider_name=provider_name,
            language=language,
            force_update=force_update,
        )

    async def analyze_overall_insight(
        self,
        *,
        user_id: str,
        timezone: Optional[str] = None,
        provider_name: Optional[str] = "",
        language: Optional[str] = "en-US",
        force_update: bool = False,
    ) -> Dict[str, Any]:
        return await self._analyze(
            prompt_template=OVERALL_INSIGHT_PROMPT_TEMPLATE,
            session_prefix=_OVERALL_SESSION_PREFIX,
            stage_prefix="overall",
            app_id=self.APP_ID_OVERALL,
            user_id=user_id,
            timezone=timezone,
            provider_name=provider_name,
            language=language,
            force_update=force_update,
        )

    async def _analyze(
        self,
        *,
        prompt_template: str,
        session_prefix: str,
        stage_prefix: str,
        app_id: str,
        user_id: str,
        timezone: Optional[str],
        provider_name: Optional[str],
        language: Optional[str],
        force_update: bool,
    ) -> Dict[str, Any]:
        target_language = (language or "en-US").strip()
        request_id = str(uuid.uuid4())
        session_id = _stable_session_id(session_prefix, user_id)
        started_at = time.perf_counter()

        log_info(
            "REQ",
            "request",
            domain=stage_prefix,
            user_id=user_id,
            request_id=request_id,
            session_id=session_id,
            timezone=timezone,
            language=target_language,
            provider_name=provider_name or "",
            force_update=force_update,
            app_id=app_id,
            flow=self.FLOW,
        )

        try:
            hidden_message = prompt_template.format(
                target_language=target_language,
                max_chars=max_chars,
            )
        except Exception as e:
            log_exception("ERR", "prompt format failed", domain=stage_prefix, err=repr(e))
            return self._err(user_id, f"prompt_format_failed: {e!r}", request_id, session_id)

        log_info(
            "PUB",
            "preparing hidden prompt",
            domain=stage_prefix,
            user_id=user_id,
            request_id=request_id,
            session_id=session_id,
            hidden_len=len(hidden_message),
        )

        chat_msg = ChatMessage(
            message=hidden_message,
            request_id=request_id,
            session_id=session_id,
            user_id=user_id,
            flow=self.FLOW,
            timezone=timezone,
            app_id=app_id,
            quick_reply=None,
            persist_history=False,
        )

        log_info(
            "PUB",
            "calling chat_service.send_message",
            domain=stage_prefix,
            user_id=user_id,
            request_id=request_id,
            session_id=session_id,
            flow=self.FLOW,
            app_id=app_id,
        )

        try:
            await self.chat_service.send_message(chat_msg)
        except HTTPException as e:
            log_error(
                "ERR",
                "publish HTTPException",
                domain=stage_prefix,
                user_id=user_id,
                request_id=request_id,
                session_id=session_id,
                status=e.status_code,
                detail=e.detail,
            )
            return self._err(
                user_id,
                f"publish_failed: [{e.status_code}] {e.detail}",
                request_id,
                session_id,
            )
        except Exception as e:
            log_exception(
                "EXC",
                "publish unexpected",
                domain=stage_prefix,
                user_id=user_id,
                request_id=request_id,
                session_id=session_id,
            )
            return self._err(user_id, f"publish_failed: {e!r}", request_id, session_id)

        log_info(
            "PUB",
            "sent ok",
            domain=stage_prefix,
            user_id=user_id,
            request_id=request_id,
            session_id=session_id,
        )

        try:
            redis_body = await self.correlation.await_response(
                user_id=user_id,
                session_id=session_id,
                request_id=request_id,
            )
        except InsightTimeoutError as e:
            log_error(
                "ERR",
                "await_response TIMEOUT",
                domain=stage_prefix,
                user_id=user_id,
                request_id=request_id,
                session_id=session_id,
                error=str(e),
            )
            return self._err(user_id, str(e), request_id, session_id)

        raw_response = redis_body.get("response") or ""
        prose = raw_response.strip()
        log_info(
            "VAL",
            "validate prose",
            domain=stage_prefix,
            user_id=user_id,
            request_id=request_id,
            session_id=session_id,
            response_len=len(raw_response),
            trimmed_len=len(prose),
            language=target_language,
        )

        if not prose:
            log_error(
                "ERR",
                "empty chat response",
                domain=stage_prefix,
                user_id=user_id,
                request_id=request_id,
                session_id=session_id,
            )
            return self._err(
                user_id, "empty_chat_response", request_id, session_id
            )

        if len(prose) < PROSE_MIN_REASONABLE_LEN:
            log_warn(
                "WRN",
                "suspiciously short prose",
                domain=stage_prefix,
                user_id=user_id,
                request_id=request_id,
                session_id=session_id,
                trimmed_len=len(prose),
                language=target_language,
            )
        if len(prose) > PROSE_MAX_REASONABLE_LEN:
            log_warn(
                "WRN",
                "unusually long prose",
                domain=stage_prefix,
                user_id=user_id,
                request_id=request_id,
                session_id=session_id,
                trimmed_len=len(prose),
                language=target_language,
            )

        elapsed_sec = round(time.perf_counter() - started_at, 3)
        preview = prose[:PREVIEW_CHARS]
        log_info(
            "RET",
            "returning SUCCESS",
            domain=stage_prefix,
            user_id=user_id,
            request_id=request_id,
            session_id=session_id,
            insight_len=len(prose),
            elapsed_sec=elapsed_sec,
            preview=preview,
        )
        return {
            "status": "success",
            "user_id": user_id,
            "insight": prose,
            "error": None,
        }

    @staticmethod
    def _err(
        user_id: str, reason: str, request_id: str, session_id: str
    ) -> Dict[str, Any]:
        log_info(
            "RET",
            "returning ERROR",
            user_id=user_id,
            request_id=request_id,
            session_id=session_id,
            error_reason=reason,
        )
        return {
            "status": "error",
            "user_id": user_id,
            "insight": None,
            "error": reason,
        }

ChatDrivenInsightService = ChatDrivenHealthInsightService

__all__ = [
    "ChatDrivenHealthInsightService",
    "ChatDrivenInsightService",
]

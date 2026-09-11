import asyncio
import json
import os
from typing import Any, Dict, Optional

from . import log_debug, log_error, log_info
from .errors import InsightTimeoutError


class CorrelationManager:
    RESPONSE_KEY_TMPL: str = "chat::{user_id}::{session_id}::{request_id}"
    DONE_STATUS: str = "done"

    def __init__(
        self,
        *,
        redis_client: Any,
        timeout_sec: Optional[int] = None,
        poll_interval_sec: float = 0.3,
    ) -> None:
        self.redis_client = redis_client
        if timeout_sec is None:
            timeout_sec = int(os.getenv("CDI_TIMEOUT_SEC", "120"))
        self.timeout_sec = timeout_sec
        self.poll_interval_sec = poll_interval_sec

    def _key(self, user_id: str, session_id: str, request_id: str) -> str:
        return self.RESPONSE_KEY_TMPL.format(
            user_id=user_id, session_id=session_id, request_id=request_id
        )

    async def await_response(
        self,
        *,
        user_id: str,
        session_id: str,
        request_id: str,
    ) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        redis = self.redis_client.redis  # raw redis-py handle (sync)
        key = self._key(user_id, session_id, request_id)
        deadline = loop.time() + self.timeout_sec

        log_info(
            "COR",
            "await_response start",
            key=key,
            timeout_sec=self.timeout_sec,
            poll_interval_sec=self.poll_interval_sec,
            user_id=user_id,
            request_id=request_id,
            session_id=session_id,
        )

        attempt = 0
        elapsed_sec = 0.0
        while True:
            poll_t0 = loop.time()
            raw = await loop.run_in_executor(None, redis.get, key)
            poll_dt_ms = (loop.time() - poll_t0) * 1000.0
            parsed: Optional[Dict[str, Any]] = None
            if raw:
                try:
                    maybe = json.loads(raw)
                except (TypeError, json.JSONDecodeError):
                    maybe = None
                if isinstance(maybe, dict):
                    parsed = maybe

            elapsed_sec = loop.time() - (deadline - self.timeout_sec)
            attempt += 1

            if parsed and parsed.get("status") == self.DONE_STATUS:
                log_info(
                    "COR",
                    "await_response done",
                    key=key,
                    elapsed_sec=round(elapsed_sec, 3),
                    status=parsed.get("status"),
                    attempt=attempt,
                    poll_dt_ms=round(poll_dt_ms, 2),
                    user_id=user_id,
                    request_id=request_id,
                )
                return parsed

            if loop.time() >= deadline:
                log_error(
                    "ERR",
                    "await_response TIMEOUT",
                    key=key,
                    elapsed_sec=round(elapsed_sec, 3),
                    attempt=attempt,
                    last_poll_dt_ms=round(poll_dt_ms, 2),
                    user_id=user_id,
                    request_id=request_id,
                )
                raise InsightTimeoutError(
                    f"No chat response within {self.timeout_sec}s at {key}"
                )

            log_info(
                "TRC",
                "poll",
                key=key,
                attempt=attempt,
                elapsed_sec=round(elapsed_sec, 3),
                poll_dt_ms=round(poll_dt_ms, 2),
                present=bool(raw),
                parsed_status=(parsed.get("status") if parsed else None),
                user_id=user_id,
                request_id=request_id,
            )

            await asyncio.sleep(self.poll_interval_sec)

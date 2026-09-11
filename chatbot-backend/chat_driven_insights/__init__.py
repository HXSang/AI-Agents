import os
from typing import Any, Dict, Optional

from utils.logger import logger as _base_logger

CDI_LOG_LEVEL: str = os.getenv("CDI_LOG_LEVEL", "INFO").upper()


def _fmt(stage: str, msg: str, fields: Optional[Dict[str, Any]] = None) -> str:
    if not fields:
        return f"[CDI-{stage}] {msg}"
    rendered = " ".join(
        f"{k}={_stringify(v)}" for k, v in fields.items()
    )
    return f"[CDI-{stage}] {msg} {rendered}"


def _stringify(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        return f"{round(value, 6)}"
    if isinstance(value, (dict, list, tuple)):
        try:
            return repr(value)
        except Exception:
            return "<unrepr>"
    return repr(value)


def log_info(stage: str, msg: str, **fields: Any) -> None:
    """INFO-level CDI log line. Extra fields are forwarded to Python logging."""
    _base_logger.info(_fmt(stage, msg, fields))


def log_warn(stage: str, msg: str, **fields: Any) -> None:
    """WARNING-level CDI log line."""
    _base_logger.warning(_fmt(stage, msg, fields))


def log_error(stage: str, msg: str, **fields: Any) -> None:
    """ERROR-level CDI log line. Surfaced in logs/errors_YYYYMMDD.log."""
    _base_logger.error(_fmt(stage, msg, fields))


def log_exception(stage: str, msg: str, **fields: Any) -> None:
    """ERROR-level CDI log line WITH traceback."""
    _base_logger.exception(_fmt(stage, msg, fields))


def log_debug(stage: str, msg: str, **fields: Any) -> None:
    """DEBUG-level CDI log line. Only emitted when CDI_LOG_LEVEL=DEBUG."""
    if CDI_LOG_LEVEL == "DEBUG":
        _base_logger.debug(_fmt(stage, msg, fields))


# Local import after helpers so circular-free usage works in main.py.
from .service import (  # noqa: E402
    ChatDrivenHealthInsightService,
    ChatDrivenInsightService,
)

__all__ = [
    "ChatDrivenHealthInsightService",
    "ChatDrivenInsightService",
    "log_info",
    "log_warn",
    "log_error",
    "log_exception",
    "log_debug",
    "CDI_LOG_LEVEL",
]

"""Exception types raised by the Chat-Driven Insights pipeline."""


class ChatDrivenInsightError(Exception):
    """Base class for all CDI failures. Routers can branch on this single type."""


class InsightTimeoutError(ChatDrivenInsightError):
    """Raised when the chat poll loop waits longer than the configured timeout."""


class AgentEmptyResponseError(ChatDrivenInsightError):
    """Raised when the chat response body in Redis is present but has no prose."""


class PublishError(ChatDrivenInsightError):
    """Raised when publishing the hidden chat message to AI agents fails."""

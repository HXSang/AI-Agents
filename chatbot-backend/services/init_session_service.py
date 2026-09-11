from abc import ABC, abstractmethod
from typing import Any, Dict

from models.models import InitSessionRequest


class InitSessionService(ABC):
    """Abstract interface for init_session operations."""

    @abstractmethod
    async def init_session(self, request: InitSessionRequest) -> Dict[str, Any]:
        """Queue init_session job to AI agents."""
        pass

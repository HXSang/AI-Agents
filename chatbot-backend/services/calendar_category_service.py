"""Calendar event category classification service interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class CalendarCategoryService(ABC):
    """Abstract interface for calendar event category detection."""

    @abstractmethod
    async def categorize_events(
        self,
        user_id: str,
        events: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Classify each event into an allowed category code."""
        pass

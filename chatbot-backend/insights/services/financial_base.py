"""Financial Insight Service Interface"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class FinancialInsightService(ABC):
    """Abstract interface for financial insight service."""

    @abstractmethod
    async def get_financial_insights(
        self,
        user_id: str,
        language: Optional[str] = "en-US",
        month: Optional[str] = None,
        force_update: bool = False,
    ) -> Dict[str, Any]:
        """Generate financial insights for a user."""
        pass

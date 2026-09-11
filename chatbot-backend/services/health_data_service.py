from abc import ABC, abstractmethod
from typing import Optional

from models.models import HealthDataPayload


class HealthDataService(ABC):
    """Abstract interface for health data service"""

    @abstractmethod
    async def create_health_data(self, payload: HealthDataPayload) -> bool:
        """
        Create or update health data for a user

        Args:
            payload: The health data payload

        Returns:
            bool: True if successful, False otherwise
        """
        pass

    @abstractmethod
    async def get_health_data(
        self, user_id: str, force_fresh: bool = False
    ) -> Optional[HealthDataPayload]:
        pass

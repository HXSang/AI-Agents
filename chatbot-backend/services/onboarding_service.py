from abc import ABC, abstractmethod
from typing import Optional

from models.models import ExternalAPIOnboardingPayload


class OnboardingService(ABC):
    """Abstract interface for onboarding service"""

    @abstractmethod
    async def create_onboarding_data(
        self, payload: ExternalAPIOnboardingPayload
    ) -> bool:
        """
        Create or update onboarding data for a user

        Args:
            payload: The onboarding data payload

        Returns:
            bool: True if successful, False otherwise
        """
        pass

    @abstractmethod
    async def get_onboarding_data(
        self, user_id: str
    ) -> Optional[ExternalAPIOnboardingPayload]:
        """
        Get onboarding data for a user

        Args:
            user_id: The user ID to get data for

        Returns:
            ExternalAPIOnboardingPayload or None if not found
        """
        pass

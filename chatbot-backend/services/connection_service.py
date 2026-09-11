from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from models.models import ConnectionData, ConnectionResponse


class ConnectionService(ABC):
    """Abstract interface for connection service operations"""

    @abstractmethod
    async def save_connection_data(
        self, connection_data: ConnectionData
    ) -> ConnectionResponse:
        """Save connection data for a user (health_app, calendar, email)"""
        pass

    @abstractmethod
    async def get_connection_data(
        self, user_id: str, connection_type: str
    ) -> Optional[Dict[str, Any]]:
        """Get connection data for a user"""
        pass

    @abstractmethod
    async def update_onboarding_data(
        self, user_id: str, connection_type: str, data: Dict[str, Any]
    ) -> None:
        """Update user's onboarding data with connection information"""
        pass

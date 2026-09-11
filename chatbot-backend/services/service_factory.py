from typing import Any, Dict

from insights.services.advanced_insight_service import AdvancedInsightService
from insights.services.base import InsightService
from insights.services.financial_base import FinancialInsightService
from insights.services.financial_executor import FinancialInsightServiceImpl
from insights.services.monthly_base import MonthlyInsightService
from insights.services.monthly_executor import MonthlyInsightServiceImpl
from services.calendar_category_service import CalendarCategoryService
from services.chat_service import ChatService
from services.connection_service import ConnectionService
from services.executor.calendar_category_service_impl import CalendarCategoryServiceImpl
from services.executor.chat_service_impl import ChatServiceImpl
from services.executor.connection_service_impl import ConnectionServiceImpl
from services.executor.external_api_service_impl import ExternalAPIServiceImpl
from services.executor.health_data_service_impl import HealthDataServiceImpl
from services.executor.init_session_service_impl import InitSessionServiceImpl
from services.executor.onboarding_service_impl import OnboardingServiceImpl
from services.executor.websocket_service_impl import WebSocketServiceImpl
from services.external_api_service import IExternalAPIService
from services.health_data_service import HealthDataService
from services.init_session_service import InitSessionService
from services.onboarding_service import OnboardingService
from services.websocket_service import WebSocketService


class ServiceFactory:
    """Factory for creating service instances"""

    _services: Dict[str, Any] = {}

    @classmethod
    def get_chat_service(cls) -> ChatService:
        """Get chat service instance"""
        if "chat_service" not in cls._services:
            cls._services["chat_service"] = ChatServiceImpl()
        return cls._services["chat_service"]

    @classmethod
    def get_init_session_service(cls) -> InitSessionService:
        """Get init_session service instance"""
        if "init_session_service" not in cls._services:
            impl = InitSessionServiceImpl()
            impl.set_chat_service(cls.get_chat_service())
            cls._services["init_session_service"] = impl
        return cls._services["init_session_service"]

    @classmethod
    def get_connection_service(cls) -> ConnectionService:
        """Get connection service instance"""
        if "connection_service" not in cls._services:
            cls._services["connection_service"] = ConnectionServiceImpl()
        return cls._services["connection_service"]

    @classmethod
    def get_websocket_service(cls) -> WebSocketService:
        """Get WebSocket service instance"""
        if "websocket_service" not in cls._services:
            cls._services["websocket_service"] = WebSocketServiceImpl()
        return cls._services["websocket_service"]

    @classmethod
    def get_external_api_service(cls) -> IExternalAPIService:
        """Get external API service instance"""
        if "external_api_service" not in cls._services:
            cls._services["external_api_service"] = ExternalAPIServiceImpl()
        return cls._services["external_api_service"]

    @classmethod
    def get_onboarding_service(cls) -> OnboardingService:
        """Get onboarding service instance"""
        if "onboarding_service" not in cls._services:
            cls._services["onboarding_service"] = OnboardingServiceImpl()
        return cls._services["onboarding_service"]

    @classmethod
    def get_health_data_service(cls) -> HealthDataService:
        """Get health data service instance"""
        if "health_data_service" not in cls._services:
            cls._services["health_data_service"] = HealthDataServiceImpl()
        return cls._services["health_data_service"]

    @classmethod
    def get_insight_service(cls) -> InsightService:
        """Get insight service instance (V3 Engine)"""
        if "insight_service" not in cls._services:
            cls._services["insight_service"] = AdvancedInsightService()
        return cls._services["insight_service"]

    @classmethod
    def get_financial_insight_service(cls) -> FinancialInsightService:
        """Get financial insight service instance"""
        if "financial_insight_service" not in cls._services:
            cls._services["financial_insight_service"] = FinancialInsightServiceImpl(
                cls.get_external_api_service()
            )
        return cls._services["financial_insight_service"]

    @classmethod
    def get_monthly_insight_service(cls) -> MonthlyInsightService:
        """Get monthly insight service instance"""
        if "monthly_insight_service" not in cls._services:
            cls._services["monthly_insight_service"] = MonthlyInsightServiceImpl(
                cls.get_external_api_service()
            )
        return cls._services["monthly_insight_service"]

    @classmethod
    def get_calendar_category_service(cls) -> CalendarCategoryService:
        """Get calendar category service instance"""
        if "calendar_category_service" not in cls._services:
            cls._services["calendar_category_service"] = CalendarCategoryServiceImpl()
        return cls._services["calendar_category_service"]

from typing import Optional

from clients.redis_client import RedisClient
from models.models import ExternalAPIOnboardingPayload
from services.onboarding_service import OnboardingService
from utils.logger import logger


class OnboardingServiceImpl(OnboardingService):
    """Implementation of onboarding service with Redis cache"""

    def __init__(self):
        self.redis_client = RedisClient()

    async def create_onboarding_data(
        self, payload: ExternalAPIOnboardingPayload
    ) -> bool:
        """
        Create or update onboarding data for a user in Redis cache

        Args:
            payload: The onboarding data payload

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create cache key with format: onboarding:user_id
            cache_key = f"onboarding:{payload.user_id}"

            # Convert payload to dictionary for JSON serialization
            data_dict = payload.model_dump()

            # Set default values for bedtime if None/null
            if not data_dict.get("bedtime_start") and not data_dict.get("bedtime_end"):
                data_dict["bedtime_start"] = "21:00"
                data_dict["bedtime_end"] = "05:00"
                data_dict["target_sleep_hours"] = 8.0
                logger.info(
                    f"Set default bedtime values for user_id: {payload.user_id} "
                    f"(bedtime_start=21:00, bedtime_end=05:00, target_sleep_hours=8)"
                )

            # Set default values for active hours if None/null
            if not data_dict.get("active_hours_start_time") and not data_dict.get(
                "active_hours_end_time"
            ):
                data_dict["active_hours_start_time"] = "09:00"
                data_dict["active_hours_end_time"] = "17:00"
                data_dict["target_active_hours"] = 8.0
                logger.info(
                    f"Set default active hours values for user_id: {payload.user_id} "
                    f"(active_hours_start_time=09:00, active_hours_end_time=17:00, target_active_hours=8)"
                )

            # Store in Redis with JSON serialization (24 hours TTL)
            await self.redis_client.set_data(cache_key, data_dict, expire=86400)

            logger.info(
                f"Successfully stored onboarding data for user_id: {payload.user_id}"
            )
            return True

        except Exception as e:
            logger.error(
                f"Error storing onboarding data for user_id {payload.user_id}: {str(e)}"
            )
            return False

    async def get_onboarding_data(
        self, user_id: str
    ) -> Optional[ExternalAPIOnboardingPayload]:
        """
        Get onboarding data for a user from Redis cache

        Args:
            user_id: The user ID to get data for

        Returns:
            ExternalAPIOnboardingPayload or None if not found
        """
        try:
            # Create cache key with format: onboarding:user_id
            cache_key = f"onboarding:{user_id}"

            # Get data from Redis
            data_dict = await self.redis_client.get_data(cache_key)

            if data_dict is None:
                logger.info(f"No onboarding data found for user_id: {user_id}")
                return None

            # Create ExternalAPIOnboardingPayload from dictionary
            payload = ExternalAPIOnboardingPayload(**data_dict)

            logger.info(
                f"Successfully retrieved onboarding data for user_id: {user_id}"
            )
            return payload

        except Exception as e:
            logger.error(
                f"Error retrieving onboarding data for user_id {user_id}: {str(e)}"
            )
            return None

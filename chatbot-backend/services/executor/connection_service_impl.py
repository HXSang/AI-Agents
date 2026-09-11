from typing import Any, Dict, Optional

from clients.redis_client import RedisClient
from models.models import ConnectionData, ConnectionResponse
from services.connection_service import ConnectionService
from utils.logger import logger


class ConnectionServiceImpl(ConnectionService):
    """Implementation of connection service"""

    def __init__(self):
        self.redis_client = RedisClient()

    async def save_connection_data(
        self, connection_data: ConnectionData
    ) -> ConnectionResponse:
        """Save connection data for a user (health_app, calendar, email)"""
        try:
            logger.log_connection_data(
                connection_data.connection_type,
                connection_data.user_id,
                success=True,
                data_size=len(str(connection_data.data)),
            )

            # Store connection data in Redis cache
            cache_key = f"connection_data:{connection_data.user_id}:{connection_data.connection_type}"
            await self.redis_client.set_data(
                cache_key, connection_data.data, expire=86400
            )  # 24 hours

            # # Also store in user's onboarding data if it exists
            # await self.update_onboarding_data(
            #     connection_data.user_id,
            #     connection_data.connection_type,
            #     connection_data.data,
            # )

            logger.log_api_response(
                "POST", "/connection", 200, user_id=connection_data.user_id
            )

            return ConnectionResponse(
                status="success",
                message=f"Connection data saved for {connection_data.connection_type}",
                user_id=connection_data.user_id,
                connection_type=connection_data.connection_type,
            )

        except Exception as e:
            logger.error(f"Error saving connection data: {str(e)}")
            raise

    async def get_connection_data(
        self, user_id: str, connection_type: str
    ) -> Optional[Dict[str, Any]]:
        """Get connection data for a user"""
        try:
            cache_key = f"connection_data:{user_id}:{connection_type}"
            data = await self.redis_client.get_data(cache_key)

            logger.log_redis_operation(
                "GET", data is not None, user_id=user_id, cache_key=cache_key
            )

            return data

        except Exception as e:
            logger.error(f"Error getting connection data: {str(e)}")
            return None

    async def update_onboarding_data(
        self, user_id: str, connection_type: str, data: Dict[str, Any]
    ) -> None:
        """Update user's onboarding data with connection information"""
        try:
            onboarding_key = f"onboarding_data:{user_id}"
            existing_data = await self.redis_client.get_data(onboarding_key)

            if existing_data:
                # Update the specific connection field
                if connection_type == "health_app":
                    existing_data["health_app_data"] = data
                elif connection_type == "calendar":
                    existing_data["calendar_data"] = data
                elif connection_type == "email":
                    existing_data["email_data"] = data

                # Save updated data back to Redis
                await self.redis_client.set_data(
                    onboarding_key, existing_data, expire=86400
                )

                logger.info(
                    f"Updated onboarding data with {connection_type} data for user {user_id}"
                )
            else:
                logger.warning(f"No existing onboarding data found for user {user_id}")

        except Exception as e:
            logger.error(f"Error updating onboarding data: {str(e)}")
            raise

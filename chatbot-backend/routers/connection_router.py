from fastapi import APIRouter, HTTPException
from models.models import ConnectionData, ConnectionResponse
from services.service_factory import ServiceFactory
from utils.logger import logger

router = APIRouter(prefix="/connection", tags=["connection"])


@router.post("/", response_model=ConnectionResponse)
async def save_connection_data(connection_data: ConnectionData):
    """Save connection data for a user (health_app, calendar, email)"""
    try:
        logger.log_api_request(
            "POST",
            "/connection",
            user_id=connection_data.user_id,
            connection_type=connection_data.connection_type,
        )

        connection_service = ServiceFactory.get_connection_service()
        result = await connection_service.save_connection_data(connection_data)

        logger.log_api_response(
            "POST",
            "/connection",
            200,
            user_id=connection_data.user_id,
            connection_type=connection_data.connection_type,
        )
        return result

    except Exception as e:
        logger.error(f"Error in save_connection_data endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

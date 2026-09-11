from fastapi import APIRouter, HTTPException
from models.models import HealthDataPayload
from services.service_factory import ServiceFactory
from utils.logger import logger

router = APIRouter(prefix="/health_data", tags=["health_data"])


@router.post("/")
async def create_health_data(payload: HealthDataPayload):
    try:
        logger.info(f"Creating health data for user_id: {payload.user_id}")

        health_data_service = ServiceFactory.get_health_data_service()
        result = await health_data_service.create_health_data(payload)

        if not result:
            raise HTTPException(status_code=500, detail="Failed to create health data")
        
        invalidated = 0
        try:
            insight_service = ServiceFactory.get_insight_service()
            invalidated = await insight_service.invalidate_qa_insight_cache(
                payload.user_id
            )
            logger.info(
                f"Invalidated {invalidated} insight cache(s) after health sync "
                f"for user_id: {payload.user_id}"
            )
        except Exception as cache_err:
            logger.warning(
                f"Could not invalidate insight cache after health sync "
                f"for user_id {payload.user_id}: {cache_err}"
            )

        logger.info(f"Successfully created health data for user_id: {payload.user_id}")
        return {
            "status": "success",
            "message": "Health data created successfully",
            "user_id": payload.user_id,
            "insight_cache_invalidated": invalidated,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error creating health data for user_id {payload.user_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to create health data: {str(e)}"
        )


@router.get("/{user_id}")
async def get_health_data(user_id: str):
    """
    Get health data for a user
    """
    try:
        logger.info(f"Getting health data for user_id: {user_id}")

        health_data_service = ServiceFactory.get_health_data_service()
        data = await health_data_service.get_health_data(user_id)

        if data is None:
            logger.warning(f"No health data found for user_id: {user_id}")
            raise HTTPException(status_code=404, detail="Health data not found")

        logger.info(f"Successfully retrieved health data for user_id: {user_id}")
        return {"status": "success", "data": data}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting health data for user_id {user_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get health data: {str(e)}"
        )

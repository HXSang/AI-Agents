from fastapi import APIRouter, HTTPException
from models.models import ExternalAPIOnboardingPayload
from services.service_factory import ServiceFactory
from utils.logger import logger

router = APIRouter(prefix="/onboarding", tags=["onboarding"])


@router.post("/")
async def create_onboarding_data(payload: ExternalAPIOnboardingPayload):
    """
    Create or update onboarding data for a user
    """
    try:
        logger.info(
            f"Creating onboarding data for user_id: {payload.user_id}, payload: {payload.model_dump()}"
        )

        onboarding_service = ServiceFactory.get_onboarding_service()
        result = await onboarding_service.create_onboarding_data(payload)

        logger.info(
            f"Successfully created onboarding data for user_id: {payload.user_id}"
        )
        return {
            "status": "success",
            "message": "Onboarding data created successfully",
            "user_id": payload.user_id,
        }

    except Exception as e:
        logger.error(
            f"Error creating onboarding data for user_id {payload.user_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to create onboarding data: {str(e)}"
        )


@router.get("/{user_id}")
async def get_onboarding_data(user_id: str):
    """
    Get onboarding data for a user
    """
    try:
        logger.info(f"Getting onboarding data for user_id: {user_id}")

        onboarding_service = ServiceFactory.get_onboarding_service()
        data = await onboarding_service.get_onboarding_data(user_id)

        if data is None:
            logger.warning(f"No onboarding data found for user_id: {user_id}")
            raise HTTPException(status_code=404, detail="Onboarding data not found")

        logger.info(f"Successfully retrieved onboarding data for user_id: {user_id}")
        return {"status": "success", "data": data}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting onboarding data for user_id {user_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get onboarding data: {str(e)}"
        )

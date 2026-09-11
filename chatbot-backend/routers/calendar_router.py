"""Calendar event category router."""

from fastapi import APIRouter, HTTPException
from models.models import (
    CalendarCategorizeRequest,
    CalendarCategorizeResponse,
    CalendarCategoryResult,
)
from services.service_factory import ServiceFactory
from utils.logger import logger

router = APIRouter(prefix="/calendar", tags=["calendar"])


@router.post("/categorize", response_model=CalendarCategorizeResponse)
async def categorize_calendar_events(payload: CalendarCategorizeRequest):
    """
    Auto-detect calendar event categories from title and description via LLM.

    Args:
        payload: user_id plus 1-50 events with summary/description metadata.

    Returns:
        One allowed category code per event index (0-based).
    """
    try:
        logger.info(
            f"📅 Calendar categorize request for user_id={payload.user_id}, "
            f"events={len(payload.events)}"
        )

        service = ServiceFactory.get_calendar_category_service()
        events = [event.model_dump() for event in payload.events]
        result = await service.categorize_events(
            user_id=payload.user_id,
            events=events,
        )

        return CalendarCategorizeResponse(
            status=result["status"],
            user_id=result["user_id"],
            categories=[CalendarCategoryResult(**row) for row in result["categories"]],
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            f"❌ Calendar categorize failed for user_id={payload.user_id}: {exc}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to categorize calendar events: {exc}",
        )

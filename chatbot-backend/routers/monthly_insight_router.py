"""Monthly Insight Router — POST /insight/monthly."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from models.models import MonthlyInsightRequest, MonthlyInsightResponse
from services.service_factory import ServiceFactory
from utils.logger import logger

router = APIRouter(prefix="/insight", tags=["insight"])


@router.post("/monthly", response_model=MonthlyInsightResponse)
async def analyze_monthly_insight(payload: MonthlyInsightRequest):
    """Generate top-3 monthly insights (Great Job / Need Attention / Opportunity).

    Thin interface: month resolution, timezone handling and the day-1-2
    "final day" window are all owned by the service. The payload's year/month
    are forwarded as-is (None → service defaults to the current month in tz).

    Pipeline (in the service):
        BE snapshot pipeline → 3-phase rollup → 41 detectors → score / dedupe
        → top-3 (1 per category) → LLM (or template) text render
        → best-effort write to Redis (anti-rep + cache).
    """
    logger.info(
        f"📅 Monthly insight request — user={payload.user_id} "
        f"year={payload.year} month={payload.month} language={payload.language} "
        f"timezone={payload.timezone} force_update={payload.force_update}"
    )

    try:
        service = ServiceFactory.get_monthly_insight_service()
        result = await service.generate_monthly(
            profile_id=payload.user_id,  # internal var name kept; value is user_id
            year=payload.year,
            month=payload.month,
            force_update=bool(payload.force_update),
            language=payload.language,
            timezone=payload.timezone,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Monthly insight failed for user {payload.user_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate monthly insight: {e}"
        )

    if result.status == "error":
        # Engine returned a structured error (eg. snapshot API unreachable).
        raise HTTPException(
            status_code=502,
            detail=result.error or "Failed to generate monthly insight",
        )

    logger.info(f"✅ Monthly insight done — user={payload.user_id}")
    return result

"""Financial Insight Router."""

from datetime import date
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.service_factory import ServiceFactory
from utils.logger import logger

router = APIRouter(prefix="/financial", tags=["financial"])

# ─────────────────────────────────────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────────────────────────────────────


class FinancialInsightRequest(BaseModel):
    user_id: str
    language: Optional[str] = "en-US"
    # YYYY-MM  – defaults to the current month when omitted
    month: Optional[str] = None
    # When True, skip Redis cache and recompute (same idea as other insight endpoints)
    force_update: Optional[bool] = False


class FinancialInsight(BaseModel):
    message: str


class FinancialInsightResponse(BaseModel):
    status: str
    user_id: str
    month: str
    insights: List[FinancialInsight]
    has_data: bool = True
    error: Optional[str] = None


@router.post("/insights", response_model=FinancialInsightResponse)
async def get_financial_insights(payload: FinancialInsightRequest):
    try:
        user_id = payload.user_id
        language = payload.language or "en-US"
        month = payload.month or f"{date.today().year}-{date.today().month:02d}"

        financial_service = ServiceFactory.get_financial_insight_service()
        result = await financial_service.get_financial_insights(
            user_id=user_id,
            language=language,
            month=month,
            force_update=bool(payload.force_update),
        )
        if result["status"] == "error":
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Failed to generate financial insights"),
            )

        return FinancialInsightResponse(
            status=result["status"],
            user_id=result["user_id"],
            month=result["month"],
            insights=[FinancialInsight(**insight) for insight in result["insights"]],
            has_data=bool(result.get("has_data", True)),
            error=result.get("error"),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            f"❌ Error in financial insight endpoint for user {payload.user_id}: {exc}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate financial insights: {exc}",
        )

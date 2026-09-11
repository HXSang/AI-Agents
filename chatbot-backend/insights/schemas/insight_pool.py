"""Schema for the full Q&A InsightPool stored in Redis."""

from datetime import datetime
from typing import List

from pydantic import BaseModel, Field

from insights.schemas.insight_item import InsightItem

POOL_TTL_SECONDS = 1800  # 30 minutes — matches ExtractedSignals TTL


class InsightPool(BaseModel):
    user_id: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    pool_expire_at: datetime  # set at build time: generated_at + POOL_TTL_SECONDS
    health_insights: List[InsightItem] = Field(default_factory=list)
    productivity_insights: List[InsightItem] = Field(default_factory=list)
    overall_insights: List[InsightItem] = Field(default_factory=list)

    def total_count(self) -> int:
        return (
            len(self.health_insights)
            + len(self.productivity_insights)
            + len(self.overall_insights)
        )

    def by_domain(self, domain: str) -> List[InsightItem]:
        if domain == "health":
            return self.health_insights
        if domain == "productivity":
            return self.productivity_insights
        if domain == "overall":
            return self.overall_insights
        return []

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}

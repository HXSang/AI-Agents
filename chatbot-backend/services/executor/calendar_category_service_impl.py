"""Calendar event category classification service implementation."""

from typing import Any, Dict, List

from agents.llm_helper import llm_response_text, parse_json_object_from_llm_text
from agents.llm_manager import LLMManager
from agents.prompt_calendar_event_category import (
    build_calendar_event_category_detection_prompt,
)
from langchain_core.messages import HumanMessage
from services.calendar_category_service import CalendarCategoryService
from services.executor.constant import CalendarEventCategoryConstants
from utils.logger import logger


class CalendarCategoryServiceImpl(CalendarCategoryService):
    """LLM-based calendar event category detection."""

    def __init__(self) -> None:
        self._llm = LLMManager().create_model(temperature=0.1)

    async def categorize_events(
        self,
        user_id: str,
        events: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        categories: List[Dict[str, Any]] = []
        llm_payload: List[Dict[str, Any]] = []
        llm_indices: List[int] = []

        for i, event in enumerate(events):
            summary = str(event.get("summary") or "").strip()
            if not summary:
                categories.append({"index": i, "category": "OTHER"})
                continue
            llm_indices.append(i)
            llm_payload.append(
                {
                    "index": i,
                    "summary": summary,
                    "description": str(event.get("description") or ""),
                    "location": str(event.get("location") or ""),
                    "event_type": str(event.get("event_type") or ""),
                }
            )

        detected: Dict[int, str] = {}
        if llm_payload:
            detected = await self._detect_categories_map_llm(llm_payload)

        for i in llm_indices:
            raw = detected.get(i)
            category = CalendarEventCategoryConstants.resolve(raw)
            categories.append({"index": i, "category": category})

        categories.sort(key=lambda row: row["index"])

        logger.info(
            f"Calendar categorize for user_id={user_id}: "
            f"{len(events)} event(s), {len(llm_payload)} sent to LLM"
        )

        return {
            "status": "success",
            "user_id": user_id,
            "categories": categories,
        }

    async def _detect_categories_map_llm(
        self, events_payload: List[Dict[str, Any]]
    ) -> Dict[int, str]:
        """Call LLM once; return index -> UPPERCASE category (may be OTHER)."""
        if not self._llm or not events_payload:
            return {}

        allowed = sorted(CalendarEventCategoryConstants.ALLOWED)
        prompt = build_calendar_event_category_detection_prompt(events_payload, allowed)

        try:
            response = await self._llm.ainvoke([HumanMessage(content=prompt)])
            text = llm_response_text(response)
            extracted = parse_json_object_from_llm_text(text)
            out: Dict[int, str] = {}
            if isinstance(extracted, dict):
                rows = extracted.get("categories", [])
                if isinstance(rows, list):
                    for row in rows:
                        if not isinstance(row, dict):
                            continue
                        if "index" not in row or "category" not in row:
                            continue
                        try:
                            idx = int(row["index"])
                        except (TypeError, ValueError):
                            continue
                        cat = str(row["category"] or "").strip().upper() or "OTHER"
                        out[idx] = cat
            return out
        except Exception as e:
            logger.warning(f"Calendar category LLM detection failed: {e}")
            return {}

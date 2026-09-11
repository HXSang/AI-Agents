"""Tavily search provider implementation."""

import logging
from typing import Any, Optional

from langchain_tavily import TavilySearch

from app.config import settings

from .base_provider import BaseSearchProvider

logger = logging.getLogger(__name__)


class TavilyProvider(BaseSearchProvider):
    """Search provider backed by the Tavily API via LangChain."""

    def __init__(self) -> None:
        """Initialise Tavily client with settings from project config."""
        self._max_results = settings.search_max_results

    def _build_client(self, time_range: Optional[str]) -> TavilySearch:
        """Create a TavilySearch instance scoped to the given *time_range*."""
        kwargs: dict[str, Any] = {
            "max_results": self._max_results,
            "search_depth": "advanced",
            "include_answer": "advanced",
            "include_raw_content": True,
        }
        if time_range:
            kwargs["time_range"] = time_range

        return TavilySearch(**kwargs)

    async def search(
        self, query: str, query_type: str = "stable"
    ) -> list[dict[str, Any]]:
        """Query Tavily using dynamic routing based on information volatility.

        Strategy:
        - If query_type == "volatile" (Tier 1): Search in "day", then fallback to "week".
        - If query_type == "stable" (Tier 2/3): Search without time limit (None) to scan the entire web.

        Returns:
            List of result dicts with keys: title, url, content,
            published_date, score. Returns [] on any failure.
        """
        if query_type == "volatile":
            tiers_to_try = ("day", "week")
            logger.info(f"🔍 Thực thi Volatile Search (Tier 1) cho: '{query}'")
        else:
            tiers_to_try = (None,)
            logger.info(f"🔍 Thực thi Stable Search (Tier 2/3) cho: '{query}'")

        for tier in tiers_to_try:
            try:
                client = self._build_client(tier)
                raw_data = await client.ainvoke(query)
                results = raw_data.get("results", [])

                if results:
                    logger.info(
                        f"Tavily returned {len(results)} result(s) for "
                        f"query '{query}' with time_range='{tier}'"
                    )
                    return [
                        {
                            "title": res.get("title", ""),
                            "url": res.get("url", ""),
                            "content": res.get("content", ""),
                            "published_date": res.get("published_date", ""),
                            "score": res.get("score", 0.0),
                        }
                        for res in results
                    ]

                logger.info(
                    f"Tavily returned 0 results for query '{query}' "
                    f"with time_range='{tier}' — trying next tier"
                )
            except Exception as e:
                logger.error(
                    f"Tavily search failed for query '{query}' "
                    f"(time_range='{tier}'): {e}"
                )

        logger.warning(
            f"Tavily returned no results across all tiers for query: {query}"
        )
        return []

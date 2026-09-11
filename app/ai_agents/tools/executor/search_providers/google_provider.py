import logging
from typing import Any, Dict, List

import httpx

from app.config import settings

from .base_provider import BaseSearchProvider

logger = logging.getLogger(__name__)

_TIMEOUT = getattr(settings, "search_timeout_seconds", 15)


class GoogleProvider(BaseSearchProvider):

    BASE_URL = "https://customsearch.googleapis.com/customsearch/v1"

    def __init__(self):
        self.api_key = settings.google_search_api_key
        self.search_engine_id = settings.google_search_engine_id
        self.max_results = min(settings.search_max_results, 10)  # CSE max = 10

        if not self.api_key:
            logger.warning("GOOGLE_API_KEY is not set.")
        if not self.search_engine_id:
            logger.warning("GOOGLE_SEARCH_ENGINE_ID is not set.")

    async def search(self, query: str) -> List[Dict[str, Any]]:
        if not self.api_key or not self.search_engine_id:
            logger.error(
                "GoogleProvider: missing GOOGLE_API_KEY or GOOGLE_SEARCH_ENGINE_ID"
            )
            return []

        try:
            params = {
                "key": self.api_key,
                "cx": self.search_engine_id,
                "q": query,
                "num": self.max_results,
            }

            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                response = await client.get(self.BASE_URL, params=params)
                response.raise_for_status()
                data = response.json()

            results: List[Dict[str, Any]] = []

            for item in data.get("items", []):
                results.append(
                    {
                        "title": item.get("title", ""),
                        "url": item.get("link", ""),
                        "content": item.get("snippet", ""),
                    }
                )

            return results

        except Exception as e:
            logger.error(f"GoogleProvider search failed for '{query}': {e}")
            return []

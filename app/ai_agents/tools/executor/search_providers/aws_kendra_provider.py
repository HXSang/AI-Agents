import asyncio
import logging
from typing import Any, Dict, List

import boto3

from app.config import settings

from .base_provider import BaseSearchProvider

logger = logging.getLogger(__name__)

_KENDRA_TIMEOUT_SECONDS = getattr(settings, "search_timeout_seconds", 20)


class AWSKendraProvider(BaseSearchProvider):

    def __init__(self):
        self.index_id = settings.aws_kendra_index_id
        self.region = settings.aws_kendra_region
        self.max_results = settings.search_max_results

        if not self.index_id:
            logger.warning("AWS_KENDRA_INDEX_ID is not set.")

    async def search(self, query: str) -> List[Dict[str, Any]]:
        if not self.index_id:
            logger.error("AWSKendraProvider: missing AWS_KENDRA_INDEX_ID")
            return []

        def _sync_query() -> List[Dict[str, Any]]:
            client = boto3.client("kendra", region_name=self.region)

            response = client.query(
                IndexId=self.index_id,
                QueryText=query,
                PageSize=self.max_results,
            )

            results: List[Dict[str, Any]] = []

            for item in response.get("ResultItems", []):
                result_type = item.get("Type", "")

                # ANSWER type — Kendra's extracted direct answer
                if result_type == "ANSWER":
                    answer_text = (
                        item.get("AdditionalAttributes", [{}])[0]
                        .get("Value", {})
                        .get("TextWithHighlightsValue", {})
                        .get("Text", "")
                    )
                    if answer_text:
                        results.append(
                            {
                                "title": "Kendra Answer",
                                "url": item.get("DocumentURI", ""),
                                "content": answer_text,
                            }
                        )

                # DOCUMENT type — standard document result
                elif result_type == "DOCUMENT":
                    excerpt = item.get("DocumentExcerpt", {}).get("Text", "")
                    results.append(
                        {
                            "title": item.get("DocumentTitle", {}).get("Text", ""),
                            "url": item.get("DocumentURI", ""),
                            "content": excerpt,
                        }
                    )

            return results

        try:
            loop = asyncio.get_running_loop()
            return await asyncio.wait_for(
                loop.run_in_executor(None, _sync_query),
                timeout=_KENDRA_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.error(
                f"AWSKendraProvider timed out after {_KENDRA_TIMEOUT_SECONDS}s "
                f"for query '{query}'"
            )
            return []
        except Exception as e:
            logger.error(f"AWSKendraProvider search failed for '{query}': {e}")
            return []

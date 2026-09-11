"""Web search tool executor."""

from typing import Any, Optional

from langchain_core.messages import HumanMessage

from app.ai_agents.tools.executor.search_providers.tavily_provider import TavilyProvider
from app.ai_agents.tools.validate.prompts.search_prompt import (
    build_search_synthesis_prompt,
    get_agent_directive,
    get_final_response_standard,
)
from app.utils.logger import logger


class WebSearchToolExecutor:
    """Executor that handles web search requests, validates, and formats results."""

    def __init__(
        self,
        user_id: str,
        llm: Any,
        session_id: Optional[str] = None,
        today_str: str = "",
    ) -> None:
        """Initialize the web search executor."""
        self.user_id = user_id
        self.session_id = session_id
        self.today_str = today_str
        self.provider = TavilyProvider()
        self.llm = llm

    def _format_raw_results(self, raw_results: list[dict]) -> str:
        """Parse and format raw JSON results into a clean string context."""
        if not raw_results:
            return ""

        context_lines = []
        for res in raw_results:
            context_lines.append(
                f"Title: {res.get('title')}\n"
                f"URL: {res.get('url')}\n"
                f"Date: {res.get('published_date', 'Unknown')}\n"
                f"Content: {res.get('content')}\n"
                "---"
            )
        return "\n".join(context_lines)

    async def execute_search(
        self, query: str, user_lang: str, query_type: str = "stable"
    ) -> str:
        """Execute web search and orchestrate data processing."""
        logger.info(
            f"[WebSearchExecutor] user={self.user_id} | query='{query}' | type='{query_type}' | lang={user_lang}"
        )
        try:
            # Truyền query_type xuống cho provider để thực hiện Dynamic Routing
            raw_results = await self.provider.search(query, query_type)
            if not raw_results:
                return "No reliable information found for this query."

            context_str = self._format_raw_results(raw_results)

            base_prompt = build_search_synthesis_prompt(
                query, context_str, user_lang, self.today_str
            )

            standard_rules = get_final_response_standard(user_lang)

            final_prompt = f"{base_prompt}\n\n{standard_rules}"

            response = await self.llm.ainvoke([HumanMessage(content=final_prompt)])

            final_result = response.content
            if isinstance(final_result, list):
                final_result = " ".join(
                    part.get("text", "")
                    for part in final_result
                    if isinstance(part, dict)
                )

            return str(final_result) + get_agent_directive()

        except Exception as exc:
            logger.exception(f"[WebSearchExecutor] Execution error | {exc}")
            return "An error occurred while fetching and synthesizing search results."

"""Search tools for the AI agent."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from langchain_core.tools import BaseTool, tool

from app.ai_agents.tools.executor.web_search_executor import WebSearchToolExecutor
from app.ai_agents.tools.validate.prompts.search_prompt import (
    create_query_formulation_prompt,
)


def create_search_tools(
    user_id: str,
    llm: Any,
    session_id: Optional[str] = None,
    user_data: Optional[Dict[str, Any]] = None,
    timezone: Optional[str] = None,
    language: Optional[str] = None,
) -> List[BaseTool]:
    """Build and return search tools scoped to a user session."""

    tz_str = timezone or "UTC"
    current_time = datetime.now(ZoneInfo(tz_str))

    today_human = current_time.strftime("%B %d, %Y")
    day_of_week = current_time.strftime("%A")
    today_str = current_time.strftime("%Y-%m-%d")
    user_lang = language or "the same language as the user's input"

    executor = WebSearchToolExecutor(
        user_id=user_id, llm=llm, session_id=session_id, today_str=today_str
    )

    tool_description = create_query_formulation_prompt(
        today_human, day_of_week, tz_str, today_str, user_lang, user_data
    )

    @tool(description=tool_description)
    async def web_search(search_query: str, query_type: str) -> str:
        """
        Search the web and return a validated, synthesized text answer.
        query_type MUST be either 'volatile' (for prices, weather, news) or 'stable' (for people, roles, facts).
        """
        synthesized_result = await executor.execute_search(
            search_query, user_lang, query_type
        )
        return synthesized_result

    return [web_search]

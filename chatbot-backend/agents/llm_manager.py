"""LLM Manager for chatbot backend agents"""

from typing import Tuple

from config import settings
from langchain_aws import ChatBedrockConverse
from langchain_openai import ChatOpenAI


class LLMManager:
    """Factory/manager to provide LLM clients based on provider selection.

    Supported providers:
    - "bedrock": uses langchain_aws.ChatBedrockConverse
    - "openai": uses langchain_openai.ChatOpenAI
    """

    def __init__(self, provider: str = "bedrock") -> None:
        self.provider = provider.lower()

    def create_model(
        self, temperature: float = 0.1, model_id: str = "", region: str = ""
    ) -> object:
        """Create a single LLM client for the configured provider."""
        if self.provider == "openai":
            return self._create_openai_model(temperature)
        # default to bedrock
        return self._create_bedrock_model(temperature, model_id, region)

    def _create_bedrock_model(
        self, temperature: float, model_id: str = "", region: str = ""
    ) -> object:
        if not model_id:
            model_id = settings.bedrock_model_id
        if not region:
            region = settings.bedrock_model_region_name

        common = dict(model_id=model_id, region_name=region, temperature=temperature)
        llm = ChatBedrockConverse(**common)
        return llm

    def _create_openai_model(self, temperature: float) -> Tuple[object, object]:
        model = settings.openai_model
        api_key = settings.openai_api_key

        common = dict(
            model=model,
            api_key=api_key,
            temperature=temperature,
            reasoning={"effort": "low"},
        )
        llm = ChatOpenAI(**common)
        return llm

"""Shared helpers for LLM responses: text extraction, JSON parsing, and insight translation."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from agents.llm_manager import LLMManager
from agents.prompt import get_insight_translation_prompts, get_language_name
from langchain_core.messages import HumanMessage, SystemMessage
from utils.logger import logger

_translation_llm: Any = None


def llm_response_text(resp: Any) -> str:
    """Normalize LangChain AIMessage (or raw content) to plain text.

    Handles string content, Bedrock-style list blocks (text / reasoning), and dict blocks.
    """
    c = getattr(resp, "content", None) if resp is not None else None
    if c is None:
        c = resp
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts: List[str] = []
        for item in c:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "text" in item:
                    parts.append(str(item["text"]))
                elif "content" in item:
                    parts.append(str(item["content"]))
            elif isinstance(item, str):
                parts.append(item)
        return " ".join(parts).strip()
    if isinstance(c, dict):
        return str(c.get("text", c.get("content", str(c))))
    return str(c)


def parse_json_object_from_llm_text(text: str) -> Optional[Dict[str, Any]]:
    """Parse a single JSON object from model output.

    Strips ```json fences, then tries ``json.loads``. If that fails, extracts the first
    ``{...}`` span (handles leading prose). Returns ``None`` if no valid object is found.
    """
    text = (text or "").strip()
    if "```" in text:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if m:
            text = m.group(1).strip()
        else:
            parts = text.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    text = part
                    break

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    start_idx = text.find("{")
    end_idx = text.rfind("}") + 1
    if start_idx != -1 and end_idx > start_idx:
        try:
            obj = json.loads(text[start_idx:end_idx])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return None


def is_english_locale(language: Optional[str]) -> bool:
    """True when locale is an English variant (en, en-US, en-GB, ...)."""
    return (language or "en-US").lower().startswith("en")


def _get_translation_llm(fallback: Any) -> Any:
    """Dedicated low-temperature model for localization (falls back to caller's llm)."""
    global _translation_llm
    if fallback is None:
        return None
    if _translation_llm is not None:
        return _translation_llm
    try:
        _translation_llm = LLMManager(provider="bedrock").create_model(temperature=0.1)
    except Exception as e:
        logger.warning(f"⚠️ Translation LLM init failed, using caller llm: {e}")
        return fallback
    return _translation_llm


def resolve_generation_language(language: Optional[str]) -> str:
    """Language for insight generation prompts.

    Non-English user locales generate in en-US; English variants keep their locale.
    """
    lang = language or "en-US"
    if is_english_locale(lang):
        return lang
    return "en-US"


async def translate_insight_dict(
    data: Dict[str, Any],
    target_language: str,
    llm: Any,
    insight_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Translate string values in *data* to *target_language*; keys unchanged.

    English locales are returned unchanged. On LLM/parse failure, returns the original dict.
    """
    if not data or is_english_locale(target_language):
        return data

    to_translate: Dict[str, str] = {
        k: v for k, v in data.items() if isinstance(v, str) and v.strip()
    }
    if not to_translate:
        return data

    if llm is None:
        logger.warning("⚠️ translate_insight_dict: llm is None, returning original")
        return data

    target_name = get_language_name(target_language)
    system_prompt, user_prompt = get_insight_translation_prompts(
        to_translate,
        target_name,
        target_language_code=target_language,
        insight_type=insight_type,
    )
    translate_llm = _get_translation_llm(llm)

    try:
        response = await translate_llm.ainvoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        content_text = llm_response_text(response)
        parsed = parse_json_object_from_llm_text(content_text)
        if not isinstance(parsed, dict):
            logger.warning("⚠️ translate_insight_dict: failed to parse LLM JSON")
            return data

        result = dict(data)
        for key, original in to_translate.items():
            translated = parsed.get(key)
            if isinstance(translated, str) and translated.strip():
                result[key] = translated
        return result
    except Exception as e:
        logger.warning(f"⚠️ translate_insight_dict failed: {e}")
        return data


async def translate_insight_text(
    text: str,
    target_language: str,
    llm: Any,
    insight_type: Optional[str] = None,
) -> str:
    """Translate a single insight string; English locales return unchanged."""
    if not text or not text.strip() or is_english_locale(target_language):
        return text
    out = await translate_insight_dict(
        {"text": text},
        target_language,
        llm,
        insight_type=insight_type,
    )
    return str(out.get("text") or text)

"""
Qwen (DashScope) LLM integration.

This project uses DashScope's OpenAI-compatible endpoint via LangChain's ChatOpenAI.
"""

from __future__ import annotations

from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel

from src.config import Settings
from src.utils.exceptions import ConfigurationError


DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def create_qwen_chat_model(
    settings: Settings,
    *,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> BaseChatModel:
    """
    Create a chat model configured for Qwen via DashScope compatible-mode.
    """
    if not settings.dashscope_api_key:
        raise ConfigurationError(
            message="Missing DashScope API key for Qwen",
            config_key="DASHSCOPE_API_KEY",
        )
    return ChatOpenAI(
        model=settings.llm_model if model is None else model,
        api_key=settings.dashscope_api_key,
        base_url=settings.qwen_base_url or DEFAULT_QWEN_BASE_URL,
        temperature=settings.llm_temperature if temperature is None else temperature,
        max_tokens=settings.llm_max_tokens if max_tokens is None else max_tokens,
    )


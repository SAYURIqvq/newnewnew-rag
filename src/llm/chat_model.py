"""
LLM chat model factory — DeepSeek via Anthropic-compatible API.

Configure in `.env` (or shell exports):
  ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
  ANTHROPIC_AUTH_TOKEN=sk-...
  ANTHROPIC_MODEL=deepseek-chat
"""

from __future__ import annotations

import os
from typing import Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable

from src.config import Settings
from src.llm.content_utils import extract_llm_text
from src.utils.exceptions import ConfigurationError


class _NormalizedLLMWrapper(Runnable):
    """Wraps a chat model so `.content` is always a string after invoke."""

    def __init__(self, llm: BaseChatModel):
        self._llm = llm

    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        response = self._llm.invoke(input, config, **kwargs)
        return self._normalize_message(response)

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        response = await self._llm.ainvoke(input, config, **kwargs)
        return self._normalize_message(response)

    @staticmethod
    def _normalize_message(response: Any) -> Any:
        if not hasattr(response, "content"):
            return response
        text = extract_llm_text(response.content)
        if text == response.content:
            return response
        if hasattr(response, "model_copy"):
            return response.model_copy(update={"content": text})
        response.content = text
        return response

DEFAULT_ANTHROPIC_BASE_URL = "https://api.deepseek.com/anthropic"
DEFAULT_LLM_MODEL = "deepseek-chat"


def create_chat_model(
    settings: Settings,
    *,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> BaseChatModel:
    """Create ChatAnthropic pointed at DeepSeek (or any Anthropic-compatible endpoint)."""
    api_key = settings.anthropic_auth_token
    if not api_key:
        raise ConfigurationError(
            message="Missing LLM API key (set ANTHROPIC_AUTH_TOKEN in .env)",
            config_key="ANTHROPIC_AUTH_TOKEN",
        )

    base_url = settings.anthropic_base_url or DEFAULT_ANTHROPIC_BASE_URL
    model_name = model or settings.llm_model or DEFAULT_LLM_MODEL

    # LangChain reads ANTHROPIC_API_KEY; keep env in sync for downstream tools
    os.environ.setdefault("ANTHROPIC_API_KEY", api_key)
    os.environ.setdefault("ANTHROPIC_BASE_URL", base_url)

    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as e:
        raise ConfigurationError(
            message="langchain-anthropic is not installed. Run: pip install langchain-anthropic",
            config_key="ANTHROPIC_AUTH_TOKEN",
        ) from e

    kwargs = {
        "model": model_name,
        "api_key": api_key,
        "base_url": base_url,
        "temperature": settings.llm_temperature if temperature is None else temperature,
        "max_tokens": settings.llm_max_tokens if max_tokens is None else max_tokens,
    }
    llm = ChatAnthropic(**kwargs)
    return _NormalizedLLMWrapper(llm)  # type: ignore[return-value]

"""
LLM chat model factory — OpenRouter via OpenAI-compatible API.

Configure in `.env` (or shell exports):
  OPENROUTER_API_KEY=sk-or-...
  OPENAI_BASE_URL=https://openrouter.ai/api/v1
  OPENAI_MODEL=deepseek/deepseek-v4-flash
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

DEFAULT_OPENAI_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_LLM_MODEL = "deepseek/deepseek-v4-flash"


def create_chat_model(
    settings: Settings,
    *,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
) -> BaseChatModel:
    """Create ChatOpenAI pointed at OpenRouter or another OpenAI-compatible endpoint."""
    api_key = settings.anthropic_auth_token
    if not api_key:
        raise ConfigurationError(
            message="Missing LLM API key (set OPENROUTER_API_KEY in .env)",
            config_key="OPENROUTER_API_KEY",
        )

    base_url = settings.anthropic_base_url or DEFAULT_OPENAI_BASE_URL
    model_name = model or settings.llm_model or DEFAULT_LLM_MODEL

    # LangChain/OpenAI-compatible clients read OPENAI_*; keep env in sync.
    os.environ.setdefault("OPENAI_API_KEY", api_key)
    os.environ.setdefault("OPENAI_BASE_URL", base_url)

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        raise ConfigurationError(
            message="langchain-openai is not installed. Run: pip install langchain-openai",
            config_key="OPENROUTER_API_KEY",
        ) from e

    kwargs = {
        "model": model_name,
        "api_key": api_key,
        "base_url": base_url,
        "temperature": settings.llm_temperature if temperature is None else temperature,
        "max_tokens": settings.llm_max_tokens if max_tokens is None else max_tokens,
        "timeout": settings.request_timeout,
        "max_retries": 2,
    }
    if reasoning_effort is not None and "openrouter.ai" in base_url:
        kwargs["extra_body"] = {
            "reasoning": {
                "effort": reasoning_effort,
                "exclude": True,
            }
        }
    llm = ChatOpenAI(**kwargs)
    return _NormalizedLLMWrapper(llm)  # type: ignore[return-value]

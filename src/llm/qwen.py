"""Backward-compatible alias — project LLM is now DeepSeek (Anthropic-compatible API)."""

from src.llm.chat_model import create_chat_model

# Legacy name used across the codebase
create_qwen_chat_model = create_chat_model

__all__ = ["create_qwen_chat_model", "create_chat_model"]

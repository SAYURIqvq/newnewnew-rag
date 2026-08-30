"""Normalize LLM response content from provider-specific message formats."""

from typing import Any, List


def extract_llm_text(content: Any) -> str:
    """
    Convert LLM message content to a plain string.

    Some LLM providers may return:
    - str
    - list of blocks: [{"type": "thinking", ...}, {"type": "text", "text": "..."}]
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                block_type = block.get("type")
                if block_type == "text":
                    text = block.get("text")
                    if text:
                        parts.append(str(text))
                elif "text" in block and block_type != "thinking":
                    parts.append(str(block["text"]))
        if parts:
            return "\n".join(parts)
    return str(content)

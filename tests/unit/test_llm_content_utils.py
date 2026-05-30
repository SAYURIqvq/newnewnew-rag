"""Tests for DeepSeek / Anthropic block-list response normalization."""

from src.llm.content_utils import extract_llm_text


def test_extract_string_passthrough():
    assert extract_llm_text("hello") == "hello"


def test_extract_deepseek_blocks():
    content = [
        {"type": "thinking", "thinking": "internal"},
        {"type": "text", "text": "Hi"},
    ]
    assert extract_llm_text(content) == "Hi"


def test_extract_multiple_text_blocks():
    content = [
        {"type": "text", "text": "Line one"},
        {"type": "text", "text": "Line two"},
    ]
    assert extract_llm_text(content) == "Line one\nLine two"

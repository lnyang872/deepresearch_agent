"""Utilities for keeping program-owned report metadata out of LLM prose."""
from __future__ import annotations

import re

__all__ = ["strip_embedded_overall_confidence"]


# Overall confidence is computed by the application, not authored by an LLM.
# Match a complete Markdown line so ordinary discussion of confidence remains intact.
_OVERALL_CONFIDENCE_LINE = re.compile(
    r"(?im)^[^\S\r\n]*(?:[-*+][^\S\r\n]+)?(?:#{1,6}[^\S\r\n]+)?(?:\*{1,3})?[^\S\r\n]*"
    r"(?:总体|总|整体|最终)?[^\S\r\n]*"
    r"(?:置信度|overall[^\S\r\n]+confidence)[^\S\r\n]*[:：]?[^\S\r\n]*"
    r"(?:0(?:\.\d+)?|1(?:\.0+)?)[^\S\r\n]*(?:\*{1,3})?"
    r"(?:[^\S\r\n]*的原因)?[^\S\r\n]*[:：]?[^\r\n]*(?:\r?\n|$)"
)


def strip_embedded_overall_confidence(content: str) -> str:
    """Remove LLM-authored overall-confidence lines from report prose.

    The final formatter owns the single delivered confidence value. Removing
    these lines instead of parsing them prevents a model from producing a
    second, conflicting score in Chinese or English.
    """
    cleaned = _OVERALL_CONFIDENCE_LINE.sub("", content or "")
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()

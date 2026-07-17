"""Utilities for keeping program-owned report metadata out of LLM prose."""
from __future__ import annotations

import re

__all__ = [
    "derive_report_title",
    "strip_embedded_overall_confidence",
    "strip_leading_report_title",
]


# Overall confidence is computed by the application, not authored by an LLM.
# Match a complete Markdown line so ordinary discussion of confidence remains intact.
_OVERALL_CONFIDENCE_LINE = re.compile(
    r"(?im)^[^\S\r\n]*(?:[-*+][^\S\r\n]+)?(?:#{1,6}[^\S\r\n]+)?(?:\*{1,3})?[^\S\r\n]*"
    r"(?:总体|总|整体|最终)?[^\S\r\n]*"
    r"(?:置信度|overall[^\S\r\n]+confidence)[^\S\r\n]*[:：]?[^\S\r\n]*"
    r"(?:0(?:\.\d+)?|1(?:\.0+)?)[^\S\r\n]*(?:\*{1,3})?"
    r"(?:[^\S\r\n]*的原因)?[^\S\r\n]*[:：]?[^\r\n]*(?:\r?\n|$)"
)

_MARKDOWN_HEADING = re.compile(r"(?m)^\s{0,3}#{1,3}\s+(.+?)\s*$")
_MARKDOWN_DECORATION = re.compile(r"[*_`#]+")


def derive_report_title(content: str, fallback: str = "研究报告") -> str:
    """Use the report's first substantive Markdown heading as its delivered title."""
    ignored = {"研究报告", "元信息", "参考来源", "已验证结论"}
    for match in _MARKDOWN_HEADING.finditer(content or ""):
        title = _normalise_title(match.group(1))
        if title and title not in ignored and not title.startswith("研究报告："):
            return title[:80]
    return fallback


def strip_leading_report_title(content: str, title: str) -> str:
    """Remove the generated title from the body when the formatter renders it."""
    lines = (content or "").splitlines()
    while lines and (not lines[0].strip() or re.fullmatch(r"\s*---+\s*", lines[0])):
        lines.pop(0)
    if lines:
        match = re.fullmatch(r"\s{0,3}#{1,3}\s+(.+?)\s*", lines[0])
        if match and _normalise_title(match.group(1)) == title:
            lines.pop(0)
    while lines and not lines[0].strip():
        lines.pop(0)
    return "\n".join(lines).strip()


def _normalise_title(value: str) -> str:
    return _MARKDOWN_DECORATION.sub("", value).strip().strip("：: ")


def strip_embedded_overall_confidence(content: str) -> str:
    """Remove LLM-authored overall-confidence lines from report prose.

    The final formatter owns the single delivered confidence value. Removing
    these lines instead of parsing them prevents a model from producing a
    second, conflicting score in Chinese or English.
    """
    cleaned = _OVERALL_CONFIDENCE_LINE.sub("", content or "")
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()

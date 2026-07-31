"""Build and enforce an auditable assertion-to-evidence ledger for reports."""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

from ..orchestrator.schemas import AgentResult, AgentStatus

__all__ = [
    "build_evidence_ledger",
    "format_evidence_ledger",
    "validate_claims",
    "format_verified_claims",
    "render_verified_claims",
    "enforce_inline_citations",
    "build_evidence_gap_notice",
]

_CITATION = re.compile(r"\[S(\d+)\]")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+")
_HORIZONTAL_RULE = re.compile(r"^\s{0,3}(?:---+|\*\*\*+|___+)\s*$")


def build_evidence_ledger(results: list[AgentResult]) -> list[dict[str, Any]]:
    """Create one stable evidence card per source admitted by ``SourceGate``.

    Only results present in the researcher trajectory are considered. That
    trajectory already contains the source-gated version of web and paper
    search responses, so a rejected hit cannot gain a citation identifier.
    """
    cards: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    verified_text: dict[str, str] = {}

    for result in results:
        for step in result.trajectory:
            if step.get("role") != "tool" or step.get("name") != "browser":
                continue
            source_url = str(step.get("verified_source_url", "")).strip()
            text = step.get("result")
            if source_url and isinstance(text, str) and len(text.strip()) >= 200:
                verified_text[source_url] = text.strip()

    for result in results:
        if result.status != AgentStatus.SUCCESS:
            continue
        for step in result.trajectory:
            if step.get("role") != "tool" or not isinstance(step.get("result"), dict):
                continue
            response = step["result"]
            # Web and paper result collections are eligible only after the
            # deterministic source gate has marked the response as accepted.
            gate = response.get("gate", {})
            if not isinstance(gate, dict) or gate.get("status") != "accepted":
                continue
            for collection_key, url_key, excerpt_key in (
                ("results", "url", "snippet"),
                ("papers", "pdf_url", "summary"),
            ):
                for item in response.get(collection_key, []):
                    if not isinstance(item, dict):
                        continue
                    url = str(item.get(url_key, "")).strip()
                    discovery_excerpt = str(item.get(excerpt_key, "")).strip()
                    if not url or not discovery_excerpt or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    full_text = verified_text.get(url, "")
                    cards.append({
                        "citation_id": f"S{len(cards) + 1}",
                        "task_id": result.task_id,
                        "url": url,
                        "title": str(item.get("title", "")).strip() or "Untitled source",
                        "evidence": (full_text or discovery_excerpt)[:1600],
                        "evidence_type": "full_text" if full_text else "discovery",
                        "verified": bool(full_text),
                        "source_quality": str(item.get("source_quality", "standard")),
                        "relevance_score": item.get("relevance_score"),
                    })
    return cards


def format_evidence_ledger(cards: list[dict[str, Any]]) -> str:
    """Render source cards for the synthesis prompt, without unverified claims."""
    if not cards:
        return "No admitted evidence cards are available. Do not make factual claims."

    sections = []
    for card in cards:
        quality = card.get("source_quality", "standard")
        relevance = card.get("relevance_score")
        evidence_type = card.get("evidence_type", "discovery")
        score = f", relevance={relevance}" if relevance is not None else ""
        sections.append(
            f"[{card['citation_id']}] {card['title']}\n"
            f"URL: {card['url']}\n"
            f"Task: {card['task_id']} | source quality={quality}{score} | evidence={evidence_type}\n"
            f"Evidence excerpt: {card['evidence']}"
        )
    return "\n\n".join(sections)


def build_evidence_gap_notice(
    results: list[AgentResult], language: str = "zh"
) -> str:
    """Render a deterministic diagnostic when no admissible evidence exists.

    The normal citation gate deliberately removes unsupported prose. This notice
    keeps the output useful without inventing facts: it reports execution-level
    diagnostics only (subtask status, tool activity, and gate rejection counts).
    """
    total = len(results)
    successful = sum(1 for result in results if result.status == AgentStatus.SUCCESS)
    failed = sum(1 for result in results if result.status == AgentStatus.FAILED)
    timed_out = sum(1 for result in results if result.status == AgentStatus.TIMEOUT)
    tool_calls = 0
    gate_rejections: Counter[str] = Counter()
    admitted = 0
    error_steps = 0
    for result in results:
        for step in result.trajectory:
            if step.get("role") == "tool":
                tool_calls += 1
                response = step.get("result")
                if isinstance(response, dict):
                    gate = response.get("gate")
                    if isinstance(gate, dict):
                        admitted += int(gate.get("accepted", 0) or 0)
                        for reason, count in (gate.get("rejection_reasons", {}) or {}).items():
                            gate_rejections[str(reason)] += int(count or 0)
            if step.get("error"):
                error_steps += 1

    rejection_text = ", ".join(
        f"{reason}={count}" for reason, count in gate_rejections.most_common()
    ) or ("none recorded" if language == "en" else "未记录")
    if language == "en":
        return "\n\n".join([
            "## Evidence Status",
            "Evidence is insufficient: no source passed the source-admission gate, so factual claims were not retained.",
            "This is a retrieval/verification diagnostic, not an answer to the research question.",
            f"- Subtasks: total={total}, successful={successful}, failed={failed}, timed_out={timed_out}",
            f"- Tool steps: {tool_calls}; admitted source items: {admitted}; tool error steps: {error_steps}",
            f"- Source-gate rejection counts: {rejection_text}",
            "- Recommended action: retry after checking search credentials/network access, query relevance, and source URLs.",
        ])
    return "\n\n".join([
        "## 证据状态",
        "证据不足：没有来源通过来源准入门禁，因此未保留事实性结论。",
        "这是一份检索/验证诊断，不是对研究问题的答案。",
        f"- 子任务：总计={total}，成功={successful}，失败={failed}，超时={timed_out}",
        f"- 工具步骤：{tool_calls}；准入来源条目={admitted}；工具错误步骤={error_steps}",
        f"- 来源门禁拒绝统计：{rejection_text}",
        "- 建议：检查搜索凭据/网络、查询相关性和来源 URL 后重试。",
    ])


def validate_claims(raw_claims: Any, cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Accept only non-empty claims that explicitly name admitted source IDs."""
    valid_ids = {str(card["citation_id"]) for card in cards}
    if not isinstance(raw_claims, list) or not valid_ids:
        return []

    claims: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw_claims[:40]:
        if not isinstance(item, dict):
            continue
        claim = _CITATION.sub("", str(item.get("claim", ""))).strip()
        section = " ".join(str(item.get("section", "研究发现")).splitlines()).strip()
        citation_ids = item.get("citation_ids", [])
        if not isinstance(citation_ids, list):
            continue
        citations = []
        for citation_id in citation_ids:
            citation_id = str(citation_id).strip().strip("[]")
            if citation_id in valid_ids and citation_id not in citations:
                citations.append(citation_id)
        fingerprint = claim.casefold()
        if not claim or not citations or fingerprint in seen:
            continue
        seen.add(fingerprint)
        claims.append({
            "section": section[:80] or "研究发现",
            "claim": claim[:600],
            "citation_ids": citations,
        })
    return claims


def format_verified_claims(claims: list[dict[str, Any]]) -> str:
    """Format claims for a writer prompt while preserving their source binding."""
    if not claims:
        return "No verified claims are available. State that evidence is insufficient."
    return "\n".join(
        f"- [{claim['section']}] {claim['claim']} "
        f"{''.join(f'[{citation_id}]' for citation_id in claim['citation_ids'])}"
        for claim in claims
    )


def render_verified_claims(claims: list[dict[str, Any]]) -> str:
    """Deterministic fallback that cannot lose verified claims or citations."""
    if not claims:
        return "证据不足：没有可通过断言-证据校验的结论。"

    sections: dict[str, list[dict[str, Any]]] = {}
    for claim in claims:
        sections.setdefault(claim["section"], []).append(claim)
    rendered = ["## 已验证结论"]
    for section, section_claims in sections.items():
        rendered.append(f"### {section}")
        rendered.extend(
            f"- {claim['claim']} {''.join(f'[{citation_id}]' for citation_id in claim['citation_ids'])}"
            for claim in section_claims
        )
    return "\n\n".join(rendered)


def enforce_inline_citations(
    content: str, cards: list[dict[str, Any]], language: str = "zh"
) -> tuple[str, list[dict[str, Any]]]:
    """Keep only substantive Markdown blocks that cite an admitted evidence card.

    A citation is valid only in the ``[S<number>]`` form and only when that
    identifier exists in this run's ledger. Headings and layout markers are
    retained; every prose paragraph and list item requires a valid inline
    citation. The returned assertion ledger records the retained block and the
    evidence cards it relies on.
    """
    valid_ids = {str(card["citation_id"]) for card in cards}
    insufficient_evidence = (
        "Evidence is insufficient: no source passed the source-admission gate, "
        "so no verifiable factual conclusion can be retained."
        if language == "en"
        else "证据不足：本次检索没有通过准入门禁的来源，无法形成可验证的事实性结论。"
    )
    missing_citation = (
        "Evidence is insufficient: the generated content did not contain valid "
        "inline citations, so its factual claims were not retained."
        if language == "en"
        else "证据不足：生成内容没有提供可验证的行内引用，未保留事实性结论。"
    )
    if not valid_ids:
        return insufficient_evidence, []

    kept: list[str] = []
    assertions: list[dict[str, Any]] = []
    for block in re.split(r"\n\s*\n", (content or "").strip()):
        lines = [line.rstrip() for line in block.splitlines()]
        nonempty = [line for line in lines if line.strip()]
        if not nonempty:
            continue
        if all(_HEADING.match(line) or _HORIZONTAL_RULE.match(line) for line in nonempty):
            kept.append("\n".join(lines))
            continue

        cited_ids = {f"S{match}" for match in _CITATION.findall(block)} & valid_ids
        if not cited_ids:
            continue

        kept.append("\n".join(lines))
        assertions.append({
            "assertion": " ".join(line.strip() for line in nonempty),
            "citations": sorted(cited_ids, key=lambda value: int(value[1:])),
        })

    cleaned = "\n\n".join(kept).strip()
    if not assertions:
        cleaned = missing_citation
    return cleaned, assertions

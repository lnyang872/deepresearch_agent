"""Build and enforce an auditable assertion-to-evidence ledger for reports."""
from __future__ import annotations

import re
from typing import Any

from ..orchestrator.schemas import AgentResult, AgentStatus

__all__ = [
    "build_evidence_ledger",
    "format_evidence_ledger",
    "validate_claims",
    "format_verified_claims",
    "render_verified_claims",
    "enforce_inline_citations",
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
                    excerpt = str(item.get(excerpt_key, "")).strip()
                    if not url or not excerpt or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    cards.append({
                        "citation_id": f"S{len(cards) + 1}",
                        "task_id": result.task_id,
                        "url": url,
                        "title": str(item.get("title", "")).strip() or "Untitled source",
                        "evidence": excerpt[:800],
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
        score = f", relevance={relevance}" if relevance is not None else ""
        sections.append(
            f"[{card['citation_id']}] {card['title']}\n"
            f"URL: {card['url']}\n"
            f"Task: {card['task_id']} | source quality={quality}{score}\n"
            f"Evidence excerpt: {card['evidence']}"
        )
    return "\n\n".join(sections)


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
    content: str, cards: list[dict[str, Any]]
) -> tuple[str, list[dict[str, Any]]]:
    """Keep only substantive Markdown blocks that cite an admitted evidence card.

    A citation is valid only in the ``[S<number>]`` form and only when that
    identifier exists in this run's ledger. Headings and layout markers are
    retained; every prose paragraph and list item requires a valid inline
    citation. The returned assertion ledger records the retained block and the
    evidence cards it relies on.
    """
    valid_ids = {str(card["citation_id"]) for card in cards}
    if not valid_ids:
        return "证据不足：本次检索没有通过准入门禁的来源，无法形成可验证的事实性结论。", []

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
        cleaned = "证据不足：生成内容没有提供可验证的行内引用，未保留事实性结论。"
    return cleaned, assertions

"""
合成 Agent (SummarizerAgent)

将多个 SubTask 的执行结果合成为结构化的研究报告。
区别于 ResearcherAgent 的多轮 tool-calling，Summarizer 是单轮长上下文生成任务：
  - 把所有子结果按置信度排序后拼接为上下文
  - 调用 LLM 一次性生成 Markdown 格式报告
  - 提取引用来源，计算整体置信度
"""
from __future__ import annotations

import json
from typing import Any

from .base_agent import BaseAgent
from ..orchestrator.schemas import SubTask, AgentResult, AgentStatus, ResearchReport
from ..utils.report_content import strip_embedded_overall_confidence
from ..utils.evidence_ledger import (
    build_evidence_ledger,
    enforce_inline_citations,
    format_evidence_ledger,
    format_verified_claims,
    render_verified_claims,
    validate_claims,
)
from ..utils.structured_output import StructuredOutputError, generate_structured
from ..utils.tracing import trace_agent


__all__ = ["SummarizerAgent"]


class SummarizerAgent(BaseAgent):
    """合成 Agent：将子任务结果合成为最终研究报告。

    Attributes:
        max_output_tokens: 报告生成的最大 token 数（通过 policy.max_tokens 控制）。
    """

    _CLAIM_SCHEMA = {
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "section": {"type": "string"},
                        "claim": {"type": "string"},
                        "citation_ids": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["section", "claim", "citation_ids"],
                },
            },
        },
        "required": ["claims"],
    }

    def __init__(self, name: str, policy, tools: list | None = None, max_refinement_rounds: int = 2) -> None:
        super().__init__(name, policy, tools)
        self.max_refinement_rounds = min(max_refinement_rounds, 3)  # 最多 3 轮，防止失控

    @trace_agent(name="summarizer.run", tags=["agent", "summarizer"])
    async def run(self, task: SubTask, context: dict) -> AgentResult:
        """执行合成任务（含迭代自优化）。

        流程: Generate → Self-Critique → Refine（最多 N 轮）。

        Args:
            task: 通常是一个特殊的 "synthesize" 类型任务。
            context: 全局上下文，必须包含 "results" 和 "query" 键。

        Returns:
            AgentResult，output 字段为 ResearchReport 实例。
        """
        query = context.get("query", "")
        results: list[AgentResult] = context.get("results", [])

        if not results:
            report = ResearchReport(
                query=query,
                content="No sub-task results available to synthesize.",
                confidence=0.0,
            )
            return AgentResult(
                task_id=task.task_id,
                status=AgentStatus.FAILED,
                output=report,
                trajectory=[],
                token_usage=0,
                confidence=0.0,
            )

        # The source-gated trajectory is turned into stable evidence cards before
        # any report prose is generated. The model cannot invent citation IDs.
        evidence_cards = build_evidence_ledger(results)

        # Stage 1: extract atomic claims and source IDs in a constrained format.
        # Stage 2 never receives unvalidated free-form factual conclusions.
        claims, claim_tokens = await self._extract_claims(query, evidence_cards)
        trajectory = [{"role": "assistant", "stage": "claims", "content": json.dumps(claims, ensure_ascii=False)}]

        fallback_content = render_verified_claims(claims)
        content, writing_tokens = await self._synthesize_once(query, claims, evidence_cards)
        content = strip_embedded_overall_confidence(content)
        content, assertion_ledger = enforce_inline_citations(content, evidence_cards)

        # A prose writer may still omit a verified claim. In that case, delivery
        # falls back to a deterministic, fully cited claim report.
        if not self._contains_all_claims(content, claims):
            content, assertion_ledger = enforce_inline_citations(fallback_content, evidence_cards)
            trajectory.append({"role": "system", "stage": "fallback", "content": "Used verified-claim fallback."})
        trajectory.append({"role": "assistant", "stage": "report", "content": content})
        token_usage = claim_tokens + writing_tokens

        # 解析最终报告
        report = self._parse_report(query, content, results, evidence_cards, assertion_ledger)
        return AgentResult(
            task_id=task.task_id,
            status=AgentStatus.SUCCESS,
            output=report,
            trajectory=trajectory,
            token_usage=token_usage,
            confidence=report.confidence,
        )

    # ------------------------------------------------------------------
    # Iterative Refinement Helpers
    # ------------------------------------------------------------------

    async def _extract_claims(
        self, query: str, evidence_cards: list[dict]
    ) -> tuple[list[dict], int]:
        """Generate claim/source pairs before allowing any report prose."""
        if not evidence_cards:
            return [], 0
        messages = [
            {"role": "system", "content": (
                "You are an evidence analyst. Return only claims directly supported by the "
                "Evidence Ledger. Every claim must identify one or more ledger IDs in "
                "citation_ids. Do not use outside knowledge or invent source IDs."
            )},
            {"role": "user", "content": self._build_claim_prompt(query, evidence_cards)},
        ]
        old_tools = getattr(self.policy, "tools", None)
        try:
            self.policy.tools = None
            payload = generate_structured(
                self.policy, messages, self._CLAIM_SCHEMA, schema_name="evidence_claims"
            )
        except (StructuredOutputError, RuntimeError, ValueError, TypeError):
            return [], 0
        finally:
            self.policy.tools = old_tools

        claims = validate_claims(payload.get("claims"), evidence_cards)
        return claims, len(json.dumps(payload, ensure_ascii=False)) // 3

    async def _synthesize_once(
        self, query: str, claims: list[dict], evidence_cards: list[dict]
    ) -> tuple[str, int]:
        """Write a report strictly from previously validated claims."""
        if not claims:
            return render_verified_claims([]), 0
        prompt = self._build_synthesis_prompt(query, claims, evidence_cards)
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": prompt},
        ]
        try:
            old_tools = getattr(self.policy, "tools", None)
            self.policy.tools = None
            response = self.policy(messages)
            self.policy.tools = old_tools
        except RuntimeError:
            return "", 0

        content = response.get("content", "") or ""
        token_usage = len(content) // 3
        return content, token_usage

    @staticmethod
    def _contains_all_claims(content: str, claims: list[dict]) -> bool:
        normalized = " ".join((content or "").split())
        return bool(claims) and all(
            " ".join(claim["claim"].split()) in normalized for claim in claims
        )

    def _system_prompt(self) -> str:
        return (
            "You are an expert research synthesizer. "
            "Write a concise Markdown report from the supplied verified claims only. "
            "Repeat every verified claim verbatim, retaining its exact [S#] citation. "
            "You may add headings and transitions, but never add a factual claim, number, "
            "comparison, or recommendation not contained in the verified claims. "
            "Do not include an overall confidence score or bibliography."
        )

    def _build_claim_prompt(self, query: str, evidence_cards: list[dict]) -> str:
        return (
            f"# Research Question\n{query}\n\n"
            "# Evidence Ledger\n"
            f"{format_evidence_ledger(evidence_cards)}\n\n"
            "# Task\n"
            "Return a JSON object with a `claims` array. Each item must contain `section`, "
            "`claim`, and `citation_ids`. Include only atomic, directly supported claims. "
            "Use only IDs shown in the Evidence Ledger."
        )

    def _build_synthesis_prompt(
        self, query: str, claims: list[dict], evidence_cards: list[dict]
    ) -> str:
        """Build a report prompt that exposes only validated claim/source pairs."""
        return (
            f"# Research Question\n{query}\n\n"
            "# Verified Claims\n"
            f"{format_verified_claims(claims)}\n\n"
            "# Instructions\n"
            "Write the report in Chinese. Include every claim above verbatim with its exact "
            "inline citation. Organize claims by their supplied section. Do not add facts from "
            "the Evidence Ledger directly; it is provided only to resolve citation labels.\n\n"
            "# Evidence Ledger\n"
            f"{format_evidence_ledger(evidence_cards)}"
        )
        return "\n".join(parts)

    def _parse_report(
        self, query: str, content: str, results: list[AgentResult],
        evidence_cards: list[dict] | None = None, assertion_ledger: list[dict] | None = None,
    ) -> ResearchReport:
        """从 LLM 输出中解析 ResearchReport，并以执行成功率计算置信度。"""
        # Overall confidence is deterministic and application-owned. Evidence-based
        # calibration will replace this provisional execution metric separately.
        total = len(results)
        success = sum(1 for r in results if r.status == AgentStatus.SUCCESS)
        success_rate = success / max(total, 1)
        confidence = round(success_rate, 2)

        evidence_cards = evidence_cards or build_evidence_ledger(results)
        sources = [
            {
                "citation_id": card["citation_id"],
                "url": card["url"],
                "title": card["title"],
                "snippet": card["evidence"],
                "task_id": card["task_id"],
            }
            for card in evidence_cards
        ]

        # 统计实际工具调用次数（遍历所有子任务的 trajectory）
        num_searches = sum(
            len([t for t in r.trajectory if t.get("role") == "tool"])
            for r in results
        )

        return ResearchReport(
            query=query,
            content=content,
            sources=sources,
            evidence_ledger=assertion_ledger or [],
            confidence=confidence,
            num_searches=num_searches,
        )

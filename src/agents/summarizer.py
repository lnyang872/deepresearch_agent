"""
合成 Agent (SummarizerAgent)

将多个 SubTask 的执行结果合成为结构化的研究报告。
区别于 ResearcherAgent 的多轮 tool-calling，Summarizer 是单轮长上下文生成任务：
  - 把所有子结果按置信度排序后拼接为上下文
  - 调用 LLM 一次性生成 Markdown 格式报告
  - 提取引用来源，计算整体置信度
"""
from __future__ import annotations

from typing import Any

from .base_agent import BaseAgent
from ..orchestrator.schemas import SubTask, AgentResult, AgentStatus, ResearchReport
from ..utils.report_content import strip_embedded_overall_confidence
from ..utils.evidence_ledger import (
    build_evidence_ledger,
    enforce_inline_citations,
    format_evidence_ledger,
)
from ..utils.tracing import trace_agent


__all__ = ["SummarizerAgent"]


class SummarizerAgent(BaseAgent):
    """合成 Agent：将子任务结果合成为最终研究报告。

    Attributes:
        max_output_tokens: 报告生成的最大 token 数（通过 policy.max_tokens 控制）。
    """

    def __init__(self, name: str, policy, tools: list | None = None, max_refinement_rounds: int = 2) -> None:
        super().__init__(name, policy, tools)
        # Kept for constructor compatibility with existing orchestration config.
        self.max_refinement_rounds = min(max_refinement_rounds, 3)

    @trace_agent(name="summarizer.run", tags=["agent", "summarizer"])
    async def run(self, task: SubTask, context: dict) -> AgentResult:
        """执行基于研究档案的深度合成。

        流程: Deep Draft → Citation Gate → Citation Repair（按需一次）。

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

        # Sources retain stable IDs, but the writer also receives the full
        # sub-task analyses needed for a genuine deep-research synthesis.
        evidence_cards = build_evidence_ledger(results)

        draft, token_usage = await self._synthesize_once(query, results, evidence_cards)
        draft = strip_embedded_overall_confidence(draft)
        content, assertion_ledger = enforce_inline_citations(draft, evidence_cards)
        trajectory = [{"role": "assistant", "stage": "draft", "content": draft}]

        # Repair citations without reducing the research to a list of claims.
        # The repair may only add or correct citations; it cannot introduce facts.
        if self._needs_citation_repair(draft, content, evidence_cards):
            repaired, repair_tokens = await self._repair_inline_citations(
                query, draft, evidence_cards
            )
            repaired = strip_embedded_overall_confidence(repaired)
            repaired_content, repaired_ledger = enforce_inline_citations(repaired, evidence_cards)
            token_usage += repair_tokens
            if len(repaired_content) > len(content):
                content, assertion_ledger = repaired_content, repaired_ledger
            trajectory.append({"role": "assistant", "stage": "citation_repair", "content": repaired})

        trajectory.append({"role": "assistant", "stage": "report", "content": content})

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
    # Deep-research synthesis and citation repair
    # ------------------------------------------------------------------

    async def _synthesize_once(
        self, query: str, results: list[AgentResult], evidence_cards: list[dict]
    ) -> tuple[str, int]:
        """Write a full report from the complete research dossier."""
        prompt = self._build_synthesis_prompt(query, results, evidence_cards)
        try:
            old_tools = getattr(self.policy, "tools", None)
            self.policy.tools = None
            response = self.policy([
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": prompt},
            ])
        except Exception:
            return "", 0
        finally:
            self.policy.tools = old_tools

        content = response.get("content", "") or ""
        return content, len(content) // 3

    async def _repair_inline_citations(
        self, query: str, draft: str, evidence_cards: list[dict]
    ) -> tuple[str, int]:
        """Ask the writer to restore missing citations while preserving the draft."""
        if not draft or not evidence_cards:
            return draft, 0
        prompt = (
            f"# Research Question\n{query}\n\n"
            "# Draft Report\n"
            f"{draft}\n\n"
            "# Evidence Ledger\n"
            f"{format_evidence_ledger(evidence_cards)}\n\n"
            "# Required Revision\n"
            "Return the complete report in Chinese. Preserve its structure, depth, and all "
            "existing content. Do not add new factual claims. Add one or more exact inline "
            "citations such as [S1] to every substantive paragraph or factual list item, "
            "using only IDs from the Evidence Ledger."
        )
        try:
            old_tools = getattr(self.policy, "tools", None)
            self.policy.tools = None
            response = self.policy([
                {"role": "system", "content": "You are a meticulous research citation editor."},
                {"role": "user", "content": prompt},
            ])
        except Exception:
            return "", 0
        finally:
            self.policy.tools = old_tools

        content = response.get("content", "") or ""
        return content, len(content) // 3

    @staticmethod
    def _needs_citation_repair(draft: str, admitted_content: str, evidence_cards: list[dict]) -> bool:
        if not draft or not evidence_cards:
            return False
        # Do not spend a second model call when the first draft already survives
        # the citation gate substantially intact.
        return len(admitted_content) < max(1200, int(len(draft) * 0.75))

    def _system_prompt(self) -> str:
        return (
            "You are an expert deep-research synthesizer. Write a detailed Chinese Markdown "
            "research report that answers the user's question, synthesizes the full task "
            "findings, compares approaches, explains tradeoffs, and gives an actionable "
            "technical roadmap. The report should normally exceed 3000 Chinese characters. "
            "Every substantive factual paragraph or list item must include one or more inline "
            "citations in the exact form [S1] or [S1][S2], using only Evidence Ledger IDs. "
            "Prefer cards marked evidence=full_text; use discovery-only cards only when the "
            "claim is limited to their title or abstract. "
            "Analysis and recommendations are allowed when clearly grounded in cited findings. "
            "Do not include an overall confidence score or a bibliography."
        )

    def _build_synthesis_prompt(
        self, query: str, results: list[AgentResult], evidence_cards: list[dict]
    ) -> str:
        """Provide both research analysis and citable source cards to the writer."""
        ordered_results = sorted(results, key=lambda result: result.confidence, reverse=True)
        task_sections = []
        for result in ordered_results:
            status = result.status.value.upper()
            output = str(result.output or "")[:3500]
            task_sections.append(
                f"## {result.task_id} ({status}, confidence={result.confidence:.2f})\n{output}"
            )
        task_dossier = "\n\n".join(task_sections)
        return (
            f"# Research Question\n{query}\n\n"
            "# Research Dossier\n"
            f"{task_dossier}\n\n"
            "# Evidence Ledger\n"
            f"{format_evidence_ledger(evidence_cards)}\n\n"
            "# Required Report Structure\n"
            "1. Executive Summary\n"
            "2. Problem Definition and Research Scope\n"
            "3. Feasibility and Research Value\n"
            "4. Current Research Progress and Evidence\n"
            "5. Technical Options and Tradeoffs\n"
            "6. Recommended Technical Route and Experiment Plan\n"
            "7. Risks, Limitations, and Next Steps\n"
            "8. Conclusion\n\n"
            "Use the dossier to synthesize rather than merely enumerate searches. State when "
            "evidence is limited or conflicting. Cite the supporting source IDs inline."
        )

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

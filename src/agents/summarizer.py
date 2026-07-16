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
from ..utils.tracing import trace_agent


__all__ = ["SummarizerAgent"]


class SummarizerAgent(BaseAgent):
    """合成 Agent：将子任务结果合成为最终研究报告。

    Attributes:
        max_output_tokens: 报告生成的最大 token 数（通过 policy.max_tokens 控制）。
    """

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

        # ---- Round 1: Initial Synthesis ----
        content, token_usage = await self._synthesize_once(query, results)
        trajectory = [{"role": "assistant", "content": content, "round": 1}]

        if not content:
            return AgentResult(
                task_id=task.task_id,
                status=AgentStatus.FAILED,
                output=ResearchReport(query=query, content="Synthesis failed.", confidence=0.0),
                trajectory=trajectory,
                token_usage=token_usage,
                confidence=0.0,
            )

        # ---- Iterative Refinement ----
        for round_idx in range(1, self.max_refinement_rounds + 1):
            # Step: Self-Critique
            critique = await self._self_critique(content, query, results)
            if not critique or not self._has_actionable_issues(critique):
                break  # 无问题，提前终止

            # Step: Refine
            refined_content = await self._refine_report(
                current_content=content,
                critique=critique,
                query=query,
                results=results,
            )
            if not refined_content:
                break

            # Gate: 改进度量 — 拒绝退化
            if not self._is_improvement(content, refined_content, results):
                break

            content = refined_content
            trajectory.append({"role": "assistant", "content": content, "round": round_idx + 1})
            token_usage += len(content) // 3

        # Overall confidence belongs to the application. Never retain an LLM-authored
        # value in the report body, including one introduced during refinement.
        content = strip_embedded_overall_confidence(content)

        # 解析最终报告
        report = self._parse_report(query, content, results)
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

    async def _synthesize_once(self, query: str, results: list[AgentResult]) -> tuple[str, int]:
        """单次合成调用（与原有逻辑一致）。"""
        prompt = self._build_synthesis_prompt(query, results)
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

    async def _self_critique(self, content: str, query: str, results: list[AgentResult]) -> str:
        """对当前报告进行结构化自检。

        检查维度:
        1. 完整性 — 是否遗漏子任务结果
        2. 矛盾性 — 是否存在内部矛盾
        3. 深广度 — 每个部分是否有充分展开
        4. 结构 — 是否符合要求结构

        Returns:
            自检结果文本（或空字符串表示无问题）。
        """
        result_summaries = []
        for i, r in enumerate(results, 1):
            status = "SUCCESS" if r.status.value == "success" else r.status.value.upper()
            result_summaries.append(
                f"[{i}] {r.task_id} ({status}): {str(r.output)[:300]}"
            )

        critique_prompt = f"""Review the following research report draft critically. Identify any issues:

## Research Question
{query}

## Sub-task Results (should all be covered)
{chr(10).join(result_summaries)}

## Draft Report
{content}

## Review Checklist
1. **Completeness**: Are ALL sub-task results reflected? List any missing.
2. **Contradictions**: Any internal contradictions between sections?
3. **Depth**: Are sections too shallow (just one sentence)? Which need expansion?
4. **Structure**: Is the report well-structured (Executive Summary → Background → Findings → Analysis → Conclusion)?

Respond in this format:
```
ISSUES_FOUND: yes|no
MISSING_RESULTS: (list task_ids not covered, or "none")
CONTRADICTIONS: (describe or "none")
SHALLOW_SECTIONS: (list section names or "none")
STRUCTURE_ISSUES: (describe or "none")
PRIORITY: high|medium|low
```
"""

        try:
            response = self.policy([
                {"role": "system", "content": "You are a critical editor. Be honest and specific."},
                {"role": "user", "content": critique_prompt},
            ])
            return response.get("content", "") or ""
        except Exception:
            return ""

    async def _refine_report(
        self, current_content: str, critique: str, query: str, results: list[AgentResult]
    ) -> str:
        """根据自检意见重写报告。"""
        refine_prompt = self._build_synthesis_prompt(query, results) + f"""

## Previous Draft

{current_content[:4000]}

## Critique (FIX THESE ISSUES)

{critique}

## Instructions

Rewrite the ENTIRE report addressing all issues in the critique above.
Keep the same structure requirements and minimum length (3000 Chinese chars / 2000 English words).
Do not include an overall confidence score. The application adds it as metadata.
"""
        try:
            response = self.policy([
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": refine_prompt},
            ])
            return response.get("content", "") or ""
        except Exception:
            return ""

    def _has_actionable_issues(self, critique: str) -> bool:
        """检查自检是否发现了可操作问题。"""
        if not critique:
            return False
        cl = critique.lower()
        # 明确标记无问题
        if "issues_found: no" in cl or "issues_found:no" in cl:
            return False
        if "issues_found: yes" in cl or "issues_found:yes" in cl:
            return True
        # 启发式检测：有具体问题描述
        indicators = ["missing", "contradiction", "shallow", "structure", "missing_results"]
        return any(ind in cl for ind in indicators)

    def _is_improvement(self, old: str, new: str, results: list[AgentResult]) -> bool:
        """门控检查：refined 版本是否真的更好。

        标准:
        1. 长度不能显著缩水（new >= old * 0.7）
        2. 覆盖度提升：引用的子任务数不能减少
        """
        if not new or len(new) < 100:
            return False

        # 1. 长度门控
        if len(new) < len(old) * 0.7:
            return False

        # 2. 覆盖度门控：统计引用 task_id 的次数
        def _count_refs(text: str) -> set:
            refs = set()
            for r in results:
                if r.task_id in text:
                    refs.add(r.task_id)
            return refs

        old_refs = _count_refs(old)
        new_refs = _count_refs(new)
        # 新版本不能遗漏旧版本已覆盖的子任务
        return len(new_refs) >= len(old_refs)

    def _system_prompt(self) -> str:
        return (
            "You are an expert research synthesizer. "
            "Your task is to integrate multiple research findings into a coherent, well-structured report. "
            "Use Markdown formatting. Cite sources explicitly. "
            "The report body MUST be at least 3000 Chinese characters (or 2000 English words) long. "
            "Write in depth: include background, key findings, detailed analysis, comparisons, and implications. "
            "DO NOT describe what you will do — directly output the synthesized report. "
            "Do not provide an overall confidence score; the application computes it from execution metadata. "
            "End with a summary of key sources."
        )

    def _build_synthesis_prompt(self, query: str, results: list[AgentResult]) -> str:
        """构建合成 prompt，按置信度降序排列结果。"""
        sorted_results = sorted(results, key=lambda r: r.confidence, reverse=True)

        parts = [
            f"# Research Question\n{query}\n",
            f"# Sub-task Results ({len(results)} total)\n",
        ]
        for i, r in enumerate(sorted_results, 1):
            status_icon = "✓" if r.status == AgentStatus.SUCCESS else "✗"
            parts.append(
                f"## Result {i} [{status_icon}] (confidence: {r.confidence:.2f})\n"
                f"Task: {r.task_id}\n"
                f"Output:\n{r.output}\n"
            )

        parts.append(
            "\n# Instructions\n"
            "1. Directly write the synthesized report based on the findings above. Do NOT say 'I will synthesize'.\n"
            "2. The report MUST be comprehensive and detailed (at least 3000 Chinese characters or 2000 English words).\n"
            "3. Structure: Executive Summary → Background → Key Findings (with details) → Analysis → Comparisons → Implications → Conclusion.\n"
            "4. Resolve any contradictions between sources.\n"
            "5. Explicitly list all sources cited.\n"
            "6. Do not include an overall confidence score; the application adds it as metadata."
        )
        return "\n".join(parts)

    def _parse_report(self, query: str, content: str, results: list[AgentResult]) -> ResearchReport:
        """从 LLM 输出中解析 ResearchReport，并以执行成功率计算置信度。"""
        # Overall confidence is deterministic and application-owned. Evidence-based
        # calibration will replace this provisional execution metric separately.
        total = len(results)
        success = sum(1 for r in results if r.status == AgentStatus.SUCCESS)
        success_rate = success / max(total, 1)
        confidence = round(success_rate, 2)

        # 收集来源（从各个子结果的轨迹中提取）
        sources: list[dict] = []
        for r in results:
            if r.status != AgentStatus.SUCCESS:
                continue
            # 简单启发式：从 trajectory 的 tool 结果中提取 url
            for step in r.trajectory:
                if step.get("role") == "tool" and isinstance(step.get("result"), dict):
                    res = step["result"]
                    if "results" in res and isinstance(res["results"], list):
                        for item in res["results"]:
                            if isinstance(item, dict) and "url" in item:
                                sources.append({
                                    "url": item["url"],
                                    "title": item.get("title", ""),
                                    "snippet": item.get("snippet", ""),
                                    "task_id": r.task_id,
                                })
                    elif "papers" in res and isinstance(res["papers"], list):
                        for paper in res["papers"]:
                            if isinstance(paper, dict) and "pdf_url" in paper:
                                sources.append({
                                    "url": paper["pdf_url"],
                                    "title": paper.get("title", ""),
                                    "snippet": paper.get("summary", "")[:200],
                                    "task_id": r.task_id,
                                })

        # 去重
        seen = set()
        unique_sources = []
        for s in sources:
            key = s["url"]
            if key not in seen:
                seen.add(key)
                unique_sources.append(s)

        # 统计实际工具调用次数（遍历所有子任务的 trajectory）
        num_searches = sum(
            len([t for t in r.trajectory if t.get("role") == "tool"])
            for r in results
        )

        return ResearchReport(
            query=query,
            content=content,
            sources=unique_sources,
            confidence=confidence,
            num_searches=num_searches,
        )

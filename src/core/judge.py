#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/core/judge.py
================================================================================
MiMo 2.5 Pro LLM-as-Judge 统一接口。

对外接口:
    - LLMJudge.score_single(report, query, ground_truth=None) -> dict
    - LLMJudge.compare_two(report_a, report_b, query) -> dict
================================================================================
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger("judge")


class LLMJudge:
    """基于 MiMo 2.5 Pro 的 LLM-as-Judge 评审器。"""

    def __init__(self, backend: str = "mimo") -> None:
        """
        Args:
            backend: Judge 后端名称，对应 ModelRouter 注册的后端。
        """
        self.backend = backend
        self._policy = None

    def _get_policy(self):
        """惰性初始化 policy，避免在导入时触发网络请求。"""
        if self._policy is None:
            from src.models.model_router import ModelRouter
            self._policy = ModelRouter.create_backend(self.backend)
        return self._policy

    # -----------------------------------------------------------------------
    # 单篇报告深度评分
    # -----------------------------------------------------------------------
    def score_single(
        self,
        report: str,
        query: str,
        ground_truth: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        对单篇报告进行 5 维度深度评分。

        返回结构:
            {
              "overall": {"score": 7.5, "reason": "..."},
              "dimensions": {
                "factual_accuracy": {"score": 8, "reason": "..."},
                "logical_consistency": {"score": 7, "reason": "..."},
                "citation_quality": {"score": 8, "reason": "..."},
                "comprehensiveness": {"score": 7, "reason": "..."}
              },
              "average": 7.5,
              "judge_backend": "mimo"
            }
        """
        gt_section = ""
        if ground_truth:
            gt_lines = "\n".join(f"- {k}: {v}" for k, v in ground_truth.items())
            gt_section = f"期望包含的关键事实：\n{gt_lines}\n"

        prompt = f"""你是一位严谨的研究报告评审专家。请对以下研究报告进行评分。

研究问题：{query}

{gt_section}
--- 研究报告 ---
{report[:4000]}

请从以下维度评分（每项 0-10 分，10 分为最高）：
1. factual_accuracy: 事实准确性（数字、日期、人名、机构名是否正确）
2. logical_consistency: 逻辑一致性（论证是否自洽，有无矛盾）
3. citation_quality: 引用质量（来源是否可靠，引用是否充分）
4. comprehensiveness: 覆盖面（是否全面回答了研究问题的各个子维度）
5. overall: 整体质量

请输出严格 JSON 格式：
{{
  "factual_accuracy": {{"score": 分数, "reason": "简短理由"}},
  "logical_consistency": {{"score": 分数, "reason": "简短理由"}},
  "citation_quality": {{"score": 分数, "reason": "简短理由"}},
  "comprehensiveness": {{"score": 分数, "reason": "简短理由"}},
  "overall": {{"score": 分数, "reason": "简短理由"}}
}}"""

        try:
            policy = self._get_policy()
            messages = [
                {"role": "system", "content": "你是研究报告评审专家。必须输出合法 JSON，不要输出任何其他内容。"},
                {"role": "user", "content": prompt},
            ]
            resp = policy(messages)
            content = resp.get("content", "")

            result = self._extract_json(content)
            if result:
                scores = [
                    v["score"]
                    for v in result.values()
                    if isinstance(v, dict) and "score" in v
                ]
                avg = sum(scores) / len(scores) if scores else 0.0
                dimensions = {k: v for k, v in result.items() if k != "overall"}
                overall = result.get("overall", {"score": avg, "reason": ""})
                return {
                    "overall": overall,
                    "dimensions": dimensions,
                    "average": avg,
                    "judge_backend": self.backend,
                }
        except Exception as e:
            logger.warning(f"MiMo Judge 单篇评分失败: {e}")
            return {"error": str(e), "judge_backend": self.backend}

        return {"error": "无法解析 MiMo Judge 输出", "judge_backend": self.backend}

    # -----------------------------------------------------------------------
    # 两篇报告 head-to-head 对比
    # -----------------------------------------------------------------------
    def compare_two(
        self,
        report_a: str,
        report_b: str,
        query: str,
        ground_truth: dict[str, Any] | None = None,
        expected_topics: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        对两份报告做 head-to-head 对比评分。

        返回结构:
            {
              "comprehensiveness": {"A": 4, "B": 5, "reason": "..."},
              "accuracy": {"A": 3, "B": 4, "reason": "..."},
              "structure": {"A": 4, "B": 4, "reason": "..."},
              "sources": {"A": 3, "B": 5, "reason": "..."},
              "judge_backend": "mimo"
            }
        """
        gt_lines = "\n".join(f"- {key}: {value}" for key, value in (ground_truth or {}).items())
        topics = "、".join(expected_topics or [])
        report_a_excerpt = self._evaluation_excerpt(report_a)
        report_b_excerpt = self._evaluation_excerpt(report_b)

        prompt = f"""你是一位严谨的研究报告评审专家。请匿名对比以下两份研究报告，从 4 个维度评分（1-5分）。

报告标签 A/B 是随机分配的，不代表 baseline 或候选系统。不要根据写作风格猜测系统身份。

研究问题：{query}

期望覆盖主题：{topics or '未提供'}

核验用关键事实：
{gt_lines or '- 未提供'}

--- 报告 A ---
{report_a_excerpt}

--- 报告 B ---
{report_b_excerpt}

评分标准：
- comprehensiveness（覆盖面）：报告是否全面回答了研究问题的各个子维度
- accuracy（准确性）：报告中的事实、数据是否正确，有无明显幻觉
- structure（结构清晰度）：报告的组织结构是否合理，逻辑是否通顺
- sources（引用质量）：报告是否引用了可靠来源，引用是否充分

请输出严格 JSON 格式：
{{
  "comprehensiveness": {{"A": 分数, "B": 分数, "reason": "简短理由"}},
  "accuracy": {{"A": 分数, "B": 分数, "reason": "简短理由"}},
  "structure": {{"A": 分数, "B": 分数, "reason": "简短理由"}},
  "sources": {{"A": 分数, "B": 分数, "reason": "简短理由"}}
}}"""

        try:
            policy = self._get_policy()
            messages = [
                {"role": "system", "content": "你是研究报告评审专家。必须输出合法 JSON，不要输出任何其他内容。"},
                {"role": "user", "content": prompt},
            ]
            resp = policy(messages)
            content = resp.get("content", "")

            result = self._extract_json(content)
            if result:
                result["judge_backend"] = self.backend
                return result
        except Exception as e:
            logger.warning(f"MiMo Judge 对比评分失败: {e}")
            return {"error": str(e), "judge_backend": self.backend}

        return {"error": "无法解析 MiMo Judge 输出", "judge_backend": self.backend}

    def compare_two_race(
        self,
        report_a: str,
        report_b: str,
        query: str,
        criteria: dict[str, Any],
        reference_report: str,
    ) -> dict[str, Any]:
        """Anonymous pairwise comparison aligned with DeepResearch Bench RACE."""
        criteria_text = self._format_race_criteria(criteria, max_chars=5200)
        reference_excerpt = self._evaluation_excerpt(reference_report, max_chars=3500)
        report_a_excerpt = self._evaluation_excerpt(report_a, max_chars=5500)
        report_b_excerpt = self._evaluation_excerpt(report_b, max_chars=5500)

        prompt = f"""你是一位严谨的深度研究报告评审专家。请依据 DeepResearch Bench 的任务专属标准，匿名比较两份报告。

报告标签 A/B 已随机分配，不代表 baseline 或候选系统。不得猜测系统身份，也不得因为篇幅更长而自动给高分。

研究任务：
{query}

任务专属评判标准：
{criteria_text}

参考报告摘录仅用于理解任务应覆盖的信息范围，不要求候选报告复述其措辞，也不要因观点不同直接扣分：
--- 参考报告 ---
{reference_excerpt}

--- 报告 A ---
{report_a_excerpt}

--- 报告 B ---
{report_b_excerpt}

请对以下四个 RACE 维度分别给 A/B 打 1-5 分：
- comprehensiveness：覆盖广度与必要细节
- insight：分析深度、因果推理、权衡与不确定性处理
- instruction_following：是否完整遵守任务中的对象、范围、时间和输出要求
- readability：结构、表达、信息组织和专业可读性

请输出严格 JSON，不要输出其他文本：
{{
  "comprehensiveness": {{"A": 分数, "B": 分数, "reason": "简短理由"}},
  "insight": {{"A": 分数, "B": 分数, "reason": "简短理由"}},
  "instruction_following": {{"A": 分数, "B": 分数, "reason": "简短理由"}},
  "readability": {{"A": 分数, "B": 分数, "reason": "简短理由"}}
}}"""

        try:
            policy = self._get_policy()
            response = policy([
                {
                    "role": "system",
                    "content": "你是独立研究报告评审专家。必须输出合法 JSON。",
                },
                {"role": "user", "content": prompt},
            ])
            content = str(response.get("content", ""))
            if content.lstrip().lower().startswith("error:"):
                return {"error": content, "judge_backend": self.backend}
            result = self._extract_json(content)
            if result:
                result["judge_backend"] = self.backend
                return result
        except Exception as exc:
            logger.warning(f"DeepResearch Bench pairwise Judge failed: {exc}")
            return {"error": str(exc), "judge_backend": self.backend}
        return {"error": "无法解析 DeepResearch Bench Judge 输出", "judge_backend": self.backend}

    @staticmethod
    def _format_race_criteria(
        criteria: dict[str, Any], max_chars: int = 5200
    ) -> str:
        """Compact official criteria so both candidate reports remain in context."""
        weights = criteria.get("dimension_weight", {})
        criterions = criteria.get("criterions", {})
        labels = {
            "comprehensiveness": "comprehensiveness",
            "insight": "insight",
            "instruction_following": "instruction_following",
            "readability": "readability",
        }
        lines: list[str] = []
        for dimension, label in labels.items():
            lines.append(f"[{label}] dimension_weight={float(weights.get(dimension, 0.0)):.3f}")
            for item in criterions.get(dimension, []):
                name = str(item.get("criterion", "")).strip()
                explanation = str(item.get("explanation", "")).strip()
                item_weight = float(item.get("weight", 0.0))
                line = f"- ({item_weight:.3f}) {name}: {explanation[:180]}"
                if len("\n".join(lines + [line])) > max_chars:
                    lines.append("- [其余细则因上下文预算省略]")
                    return "\n".join(lines)
                lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _evaluation_excerpt(report: str, max_chars: int = 12000) -> str:
        """保留报告开头与结尾，避免长报告的结论和来源被静默截掉。"""
        if len(report) <= max_chars:
            return report
        head_chars = int(max_chars * 0.7)
        tail_chars = max_chars - head_chars
        return (
            report[:head_chars]
            + "\n\n[中间内容因 Judge 上下文预算省略]\n\n"
            + report[-tail_chars:]
        )

    # -----------------------------------------------------------------------
    # 内部工具：JSON 提取
    # -----------------------------------------------------------------------
    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        """从文本中提取 JSON 对象，支持多种 fallback 策略。"""
        # 策略 1: 直接找最外层 {}
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass

        # 策略 2: 找 ```json ... ``` 代码块
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

        # 策略 3: 修复常见 JSON 错误后再解析
        cleaned = text.strip()
        # 去除可能的 Markdown 标记
        cleaned = re.sub(r"^```.*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
        # 修复单引号
        cleaned = cleaned.replace("'", '"')
        # 修复 trailing comma
        cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        return None

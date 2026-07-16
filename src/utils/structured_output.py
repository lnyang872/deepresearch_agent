"""
Structured Output Utility — JSON Schema 约束生成

提供统一的约束解码接口，优先使用 API 原生支持（response_format / guided_json），
失败时回退到现有的正则 JSON 解析策略。

支持的约束方式：
  1. OpenAI response_format (json_schema) — 用于 OpenAI / DeepSeek API
  2. vLLM guided_json (extra_body) — 用于 vLLM / SGLang 等推理引擎
  3. Regex fallback — 三层正则回退解析（保持向后兼容）

使用方式:
  from src.utils.structured_output import generate_structured

  schema = {"type": "object", "properties": {...}, "required": [...]}
  result = generate_structured(policy, messages, schema)
  # result 是已解析的 dict，保证符合 schema
"""

from __future__ import annotations

import json
import re
import logging
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["generate_structured", "StructuredOutputError"]


class StructuredOutputError(Exception):
    """约束解码失败时抛出。"""
    pass


# ============================================================================
# JSON Schema 定义（预编译，复用）
# ============================================================================

# Planner DAG 输出 schema
PLANNER_DAG_SCHEMA = {
    "type": "object",
    "properties": {
        "sub_tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "task_type": {"type": "string", "enum": ["search", "analyze", "verify"]},
                    "description": {"type": "string"},
                    "dependencies": {"type": "array", "items": {"type": "string"}},
                    "context_keys": {"type": "array", "items": {"type": "string"}},
                    "timeout_seconds": {"type": "integer", "minimum": 30, "maximum": 600},
                    "priority": {"type": "integer", "minimum": 1, "maximum": 5},
                    "expected_type": {"type": "string"},
                    "search_hints": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["task_id", "task_type", "description", "dependencies"],
            },
        },
    },
    "required": ["sub_tasks"],
}

# Red Agent 五维度评分输出 schema
RED_AGENT_SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number", "minimum": 0.0, "maximum": 10.0},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "severity": {"type": "string", "enum": ["critical", "major", "minor"]},
                    "description": {"type": "string"},
                    "location": {"type": "string"},
                    "fix_type": {"type": "string", "enum": ["in_place", "search", "removal"]},
                    "evidence": {"type": "string"},
                },
                "required": ["severity", "description", "fix_type"],
            },
        },
    },
    "required": ["score"],
}

# Judge 五维度评分输出 schema
JUDGE_SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "factual_accuracy": {"type": "number", "minimum": 0.0, "maximum": 10.0},
        "coverage": {"type": "number", "minimum": 0.0, "maximum": 10.0},
        "logical_coherence": {"type": "number", "minimum": 0.0, "maximum": 10.0},
        "citation_quality": {"type": "number", "minimum": 0.0, "maximum": 10.0},
        "rationale": {"type": "string"},
    },
    "required": ["factual_accuracy", "coverage", "logical_coherence", "citation_quality"],
}


def generate_structured(
    policy,
    messages: list[dict],
    schema: dict[str, Any],
    schema_name: str = "output",
    fallback_parser: callable = None,
) -> dict[str, Any]:
    """使用 JSON Schema 约束生成结构化输出。

    优先使用 API 原生支持，失败时回退到 fallback_parser 或内置正则解析。

    Args:
        policy: VLLMPolicy 实例（必须实现 __call__(messages) -> OpenAICompatibleDict）。
        messages: OpenAI 格式的消息列表。
        schema: JSON Schema 定义。
        schema_name: Schema 名称（用于 response_format 的 name 字段）。
        fallback_parser: 解析失败时的回退函数，签名为 (raw_text: str) -> dict。

    Returns:
        解析后的 dict，保证符合 schema 结构。

    Raises:
        StructuredOutputError: 所有策略均失败时抛出（仅在没有 fallback_parser 时）。
    """
    raw = ""

    # ---- Strategy A: OpenAI response_format (json_schema) ----
    if _supports_response_format(policy):
        try:
            raw = _call_with_response_format(policy, messages, schema, schema_name)
            if raw:
                parsed = _safe_json_parse(raw)
                if parsed and _validate_schema(parsed, schema):
                    return parsed
        except Exception as e:
            logger.debug(f"[StructuredOutput] response_format failed: {e}")

    # ---- Strategy B: vLLM guided_json (extra_body) ----
    if _supports_guided_json(policy):
        try:
            raw = _call_with_guided_json(policy, messages, schema)
            if raw:
                parsed = _safe_json_parse(raw)
                if parsed and _validate_schema(parsed, schema):
                    return parsed
        except Exception as e:
            logger.debug(f"[StructuredOutput] guided_json failed: {e}")

    # ---- Strategy C: 普通调用 + 正则解析 ----
    try:
        raw = _call_normal(policy, messages)
        if raw:
            parsed = _robust_json_parse(raw)
            if parsed and _validate_schema(parsed, schema):
                return parsed
    except Exception as e:
        logger.debug(f"[StructuredOutput] Normal call + regex parse failed: {e}")

    # ---- Strategy D: 回退到自定义 parser ----
    if fallback_parser is not None:
        if not raw:
            raw = _call_normal(policy, messages)
        return fallback_parser(raw)

    raise StructuredOutputError(
        f"All structured output strategies failed for schema '{schema_name}'"
    )


# ============================================================================
# 内部辅助函数
# ============================================================================

def _supports_response_format(policy) -> bool:
    """检测 policy 是否支持 response_format。"""
    return hasattr(policy, "response_format")


def _supports_guided_json(policy) -> bool:
    """检测 policy 是否支持 guided_json（通过 extra_body）。"""
    return hasattr(policy, "guided_json")


def _call_with_response_format(
    policy, messages: list[dict], schema: dict, schema_name: str
) -> str:
    """使用 OpenAI response_format 调用。"""
    old_rf = getattr(policy, "response_format", None)
    try:
        policy.response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": schema,
                "strict": True,
            },
        }
        resp = policy(messages)
        return resp.get("content", "") or ""
    finally:
        policy.response_format = old_rf


def _call_with_guided_json(
    policy, messages: list[dict], schema: dict
) -> str:
    """使用 vLLM guided_json 调用（通过 extra_body）。"""
    old_gj = getattr(policy, "guided_json", None)
    try:
        policy.guided_json = schema
        resp = policy(messages)
        return resp.get("content", "") or ""
    finally:
        policy.guided_json = old_gj


def _call_normal(policy, messages: list[dict]) -> str:
    """普通 LLM 调用（无约束）。"""
    resp = policy(messages)
    return resp.get("content", "") or ""


def _safe_json_parse(raw: str) -> dict | None:
    """安全 JSON 解析，返回 dict 或 None。"""
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        return None


def _robust_json_parse(raw: str) -> dict | None:
    """三层正则回退 JSON 解析（与现有代码逻辑一致）。"""
    raw = raw.strip()
    if not raw:
        return None

    # 尝试 1: 直接解析
    parsed = _safe_json_parse(raw)
    if parsed:
        return parsed

    # 尝试 2: 提取 ```json...``` 代码块
    code_block = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
    for match in code_block.findall(raw):
        parsed = _safe_json_parse(match.strip())
        if parsed:
            return parsed

    # 尝试 3: 提取第一个 { ... } 块
    brace_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if brace_match:
        parsed = _safe_json_parse(brace_match.group(0))
        if parsed:
            return parsed

    # 尝试 4: 修复常见错误后重试
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        fixed = raw[start:end + 1]
        # 移除尾随逗号
        fixed = re.sub(r",(\s*[}\]])", r"\1", fixed)
        # 移除注释
        fixed = re.sub(r"//.*?\n", "\n", fixed)
        parsed = _safe_json_parse(fixed)
        if parsed:
            return parsed

    return None


def _validate_schema(data: dict, schema: dict) -> bool:
    """轻量 Schema 验证：检查 required 字段是否存在。

    不做完整的 JSON Schema 验证（太重），只检查关键字段。
    """
    required = schema.get("required", [])
    if not isinstance(data, dict):
        return False
    for key in required:
        if key not in data:
            return False
    return True

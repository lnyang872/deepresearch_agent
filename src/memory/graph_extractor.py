"""
GraphRAG Entity-Relation Extractor — 从 MemoryEntry 中抽取结构化知识三元组

使用 LLM 从 claim 文本中抽取实体和关系，构建知识图谱。
设计为轻量级：每个 entry 只抽取关键实体（最多 5 个）和关系（最多 8 个），
避免图谱膨胀。抽取失败时静默降级（不影响主流程）。
"""

from __future__ import annotations

import json
import logging
import re
import hashlib
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["GraphExtractor"]

# 实体-关系抽取 Prompt（轻量，单 claim 限制 800 chars）
EXTRACTION_PROMPT = """Extract key entities and their relationships from the following claim.

## Claim
{claim}

## Instructions
1. Identify up to 5 key entities (people, companies, products, metrics, dates, places, concepts).
2. For each pair of related entities, describe their relationship using a short predicate (e.g., "reported_revenue", "acquired", "located_in", "exceeds").
3. Output up to 8 relationships.

## Output Format (JSON only, no markdown)
{{
  "entities": [
    {{"name": "string", "type": "person|organization|metric|date|place|concept|other"}}
  ],
  "relations": [
    {{"subject": "entity_name", "predicate": "short_verb_phrase", "object": "entity_name"}}
  ]
}}
"""


class GraphExtractor:
    """LLM 驱动的实体-关系三元组抽取器。

    从 MemoryEntry 的 claim 文本中抽取 (subject, predicate, object) 三元组，
    并生成实体节点。所有抽取操作通过 VLLMPolicy 完成。

    Attributes:
        policy: VLLMPolicy 实例（用于 LLM 调用）。
        enabled: 是否启用图谱抽取（可配置关闭）。
    """

    def __init__(self, policy=None, enabled: bool = True):
        self.policy = policy
        self.enabled = enabled
        self._extraction_count = 0
        self._failure_count = 0

    def extract(self, claim: str) -> dict[str, Any]:
        """从 claim 文本中抽取实体和关系。

        Args:
            claim: MemoryEntry.claim 文本（已截断至 800 chars）。

        Returns:
            {"entities": [...], "relations": [...]} 或空字典（失败时）。
        """
        if not self.enabled or not self.policy or not claim.strip():
            return {}

        # 截断过长的 claim（节省 LLM 调用成本）
        truncated = claim[:800] if len(claim) > 800 else claim

        try:
            prompt = EXTRACTION_PROMPT.format(claim=truncated)
            messages = [
                {"role": "system", "content": "You are a knowledge graph extractor. Output JSON only."},
                {"role": "user", "content": prompt},
            ]
            response = self.policy(messages)
            raw = response.get("content", "") or ""
            data = self._parse_extraction(raw)
            self._extraction_count += 1
            return data
        except Exception as e:
            self._failure_count += 1
            logger.debug(f"[GraphExtractor] Extraction failed: {e}")
            return {}

    def extract_from_entry(self, entry) -> dict[str, Any]:
        """从 MemoryEntry 中抽取并生成 entity_id + relation 记录。

        Args:
            entry: MemoryEntry 实例（必须有 claim 和 entry_id 字段）。

        Returns:
            {
                "entities": [{"entity_id": str, "name": str, "type": str}],
                "relations": [{"subject_id": str, "predicate": str, "object_id": str, "source_entry_id": str}],
            }
        """
        claim = getattr(entry, "claim", "") or ""
        entry_id = getattr(entry, "entry_id", "") or ""
        result = self.extract(claim)

        if not result:
            return {"entities": [], "relations": []}

        entities = result.get("entities", [])
        relations = result.get("relations", [])

        # 为每个实体生成稳定的 entity_id
        entity_name_to_id: dict[str, str] = {}
        processed_entities = []

        for e in entities:
            name = e.get("name", "").strip()
            if not name or len(name) < 2:
                continue
            eid = self._make_entity_id(name)
            entity_name_to_id[name] = eid
            processed_entities.append({
                "entity_id": eid,
                "name": name,
                "type": e.get("type", "other"),
            })

        # 将关系中的实体名映射为 entity_id
        processed_relations = []
        for r in relations:
            subj_name = r.get("subject", "").strip()
            obj_name = r.get("object", "").strip()
            if not subj_name or not obj_name:
                continue
            subj_id = entity_name_to_id.get(subj_name)
            obj_id = entity_name_to_id.get(obj_name)
            # 如果实体不在抽取列表中，生成临时 ID
            if not subj_id:
                subj_id = self._make_entity_id(subj_name)
            if not obj_id:
                obj_id = self._make_entity_id(obj_name)

            processed_relations.append({
                "subject_id": subj_id,
                "predicate": r.get("predicate", "related_to"),
                "object_id": obj_id,
                "source_entry_id": entry_id,
            })

        return {
            "entities": processed_entities,
            "relations": processed_relations,
        }

    @staticmethod
    def _make_entity_id(name: str) -> str:
        """生成稳定的 entity_id（基于名称的 MD5 哈希）。"""
        h = hashlib.md5(name.lower().strip().encode("utf-8")).hexdigest()[:12]
        return f"ent_{h}"

    def _parse_extraction(self, raw: str) -> dict[str, Any]:
        """解析 LLM 的 JSON 输出（与项目全局模式一致的健壮解析）。"""
        raw = raw.strip()
        if not raw:
            return {}

        # 尝试直接解析
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # 尝试提取 ```json 代码块
        code_match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
        if code_match:
            try:
                return json.loads(code_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # 尝试提取第一个 {...} 块
        brace_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        return {}

    @property
    def stats(self) -> dict:
        """返回抽取统计。"""
        return {
            "extraction_count": self._extraction_count,
            "failure_count": self._failure_count,
            "success_rate": (
                self._extraction_count / max(self._extraction_count + self._failure_count, 1)
            ),
        }

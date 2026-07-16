"""
GraphRAG Retriever — 图谱增强的双通道语义检索

在向量检索基础上叠加知识图谱遍历，实现：
  1. 向量通道：与现有逻辑一致（cosine similarity top-k）
  2. 图谱通道：种子实体 → 1-hop 邻居展开 → 社区感知评分
  3. 双通道合并：去重后按组合分数排序

设计决策：
  - 图谱通道不替代向量通道，而是作为补充（ORM 双通道）。
  - 1-hop 遍历限制防止图谱爆炸。
  - 相关性评分 = 向量相似度 × 0.7 + 图谱结构分 × 0.3。
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .long_term import LongTermMemory

logger = logging.getLogger(__name__)

__all__ = ["GraphRetriever"]


class GraphRetriever:
    """图谱增强检索器。

    接收向量通道的种子条目，沿知识图谱展开获取相关条目。

    Attributes:
        long_term: LongTermMemory 实例（提供图谱表的 CRUD 访问）。
        max_graph_expansion: 单次图谱展开的最大条目数。
    """

    def __init__(self, long_term: "LongTermMemory", max_graph_expansion: int = 20):
        self.long_term = long_term
        self.max_graph_expansion = max_graph_expansion

    def retrieve(
        self,
        seed_entry_ids: list[str],
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """基于种子条目沿图谱扩展检索。

        流程：
        1. 对每个种子条目，找到其关联的实体。
        2. 对每个实体，沿 relation 做 1-hop 邻居展开。
        3. 获取邻居实体关联的条目。
        4. 去重，按社区感知评分排序。

        Args:
            seed_entry_ids: 向量通道返回的种子条目 ID 列表。
            top_k: 返回的最大条目数。

        Returns:
            [(entry_id, graph_score), ...] 按分数降序排列。
        """
        if not seed_entry_ids:
            return []

        # Step 1: 收集种子实体（去重）
        seed_entities: set[str] = set()
        entry_entity_map: dict[str, set[str]] = {}  # entry_id -> {entity_id}

        for eid in seed_entry_ids:
            entities = self.long_term.get_entities_for_entry(eid)
            if entities:
                seed_entities.update(entities)
                entry_entity_map[eid] = set(entities)

        if not seed_entities:
            return []

        # Step 2: 1-hop 邻居展开
        neighbor_entities: dict[str, set[str]] = {}  # entity -> {neighbor_entity}
        entity_entry_map: dict[str, set[str]] = {}   # entity -> {entry_id}

        for entity_id in seed_entities:
            relations = self.long_term.get_entity_relations(entity_id)
            neighbors = set()
            related_entries = set()
            for rel in relations:
                neighbor = rel.get("neighbor_entity", "")
                if neighbor and neighbor != entity_id:
                    neighbors.add(neighbor)
                # 收集关系直接关联的条目
                source_eid = rel.get("source_entry_id", "")
                if source_eid:
                    related_entries.add(source_eid)

            neighbor_entities[entity_id] = neighbors
            if related_entries:
                entity_entry_map[entity_id] = related_entries

        # 对邻居实体，也查找它们关联的条目
        all_neighbors = set()
        for neighbors in neighbor_entities.values():
            all_neighbors.update(neighbors)

        for neighbor_entity in all_neighbors:
            if neighbor_entity not in entity_entry_map:
                related = set()
                relations = self.long_term.get_entity_relations(neighbor_entity)
                for rel in relations:
                    source_eid = rel.get("source_entry_id", "")
                    if source_eid:
                        related.add(source_eid)
                entity_entry_map[neighbor_entity] = related

        # Step 3: 计算图谱结构分
        scored_entries: dict[str, float] = {}

        for entity_id, entries in entity_entry_map.items():
            for eid in entries:
                if eid in seed_entry_ids:
                    continue  # 跳过已有的种子条目（去重）
                # 社区感知：实体连接的条目数越多 → 分数越高
                community_size = len(entries)
                # 如果实体是邻居（非种子）→ 分数稍低
                is_neighbor = entity_id in all_neighbors
                base_score = 0.5 if is_neighbor else 1.0
                # 连接到多个种子的实体 → 桥接分
                bridge_bonus = 0.0
                n_seed_connections = sum(
                    1 for seed_ent, neighbors in neighbor_entities.items()
                    if entity_id in neighbors
                )
                bridge_bonus = min(n_seed_connections * 0.2, 0.6)

                graph_score = base_score * min(community_size / 5.0, 1.0) + bridge_bonus
                scored_entries[eid] = max(scored_entries.get(eid, 0.0), graph_score)

        # Step 4: 排序并截断
        sorted_entries = sorted(scored_entries.items(), key=lambda x: x[1], reverse=True)
        return sorted_entries[:min(top_k, self.max_graph_expansion)]

    def merge_scores(
        self,
        vector_results: list[tuple[str, float]],
        graph_results: list[tuple[str, float]],
        vector_weight: float = 0.7,
    ) -> list[tuple[str, float]]:
        """合并向量通道和图谱通道的检索结果。

        合并策略：
        - 向量独有项保留原分 × vector_weight
        - 图谱独有项保留原分 × (1 - vector_weight)
        - 双通道共有项取加权平均

        Args:
            vector_results: [(entry_id, similarity_score), ...]
            graph_results: [(entry_id, graph_score), ...]
            vector_weight: 向量通道权重（默认 0.7）。

        Returns:
            [(entry_id, combined_score), ...] 按分数降序，最多 50 条。
        """
        combined: dict[str, float] = {}

        for eid, sim in vector_results:
            combined[eid] = sim * vector_weight

        for eid, gscore in graph_results:
            graph_contrib = gscore * (1.0 - vector_weight)
            if eid in combined:
                combined[eid] = max(combined[eid], (combined[eid] / vector_weight + gscore) / 2)
            else:
                combined[eid] = graph_contrib

        sorted_combined = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return sorted_combined[:50]

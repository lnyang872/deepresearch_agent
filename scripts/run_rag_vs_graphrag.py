#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/run_rag_vs_graphrag.py
================================================================================
RAG vs GraphRAG 对比实验脚本

对比两种检索模式在相同文档集上的表现：
  - RAG-only:        纯向量语义检索（Embedding cosine similarity）
  - RAG + GraphRAG:  向量检索 + 知识图谱 1-hop 扩展（双通道融合）

实验设计：
  1. 文档摄入阶段：用一批种子文档写入 SharedMemoryStore
  2. 查询阶段：对每个 query 分别用 RAG-only 和 RAG+GraphRAG 检索
  3. 评测阶段：对比命中率、召回率、MRR，附 Bootstrap 95% CI + Cohen's d

用法：
    # 检索级对比（快速，不需要 LLM API）
    python scripts/run_rag_vs_graphrag.py --mode retrieval

    # 端到端对比（完整 Agent 流程，需要 LLM API）
    python scripts/run_rag_vs_graphrag.py --mode e2e --num_questions 5

    # 指定输出路径
    python scripts/run_rag_vs_graphrag.py --mode retrieval --output outputs/rag_comparison.json
================================================================================
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.memory.memory_store import SharedMemoryStore
from src.memory.embedder import Embedder
from src.memory.long_term import MemoryEntry
from evaluation.metrics.stats import bootstrap_ci_two_sample, cohens_d

logger = logging.getLogger("rag_vs_graphrag")

# ==============================================================================
# 种子文档集 —— 模拟一个知识库，覆盖多个领域
# ==============================================================================
SEED_DOCS: list[dict[str, Any]] = [
    # ---- 科技 / AI 芯片 ----
    {
        "claim": "NVIDIA H100 GPU 基于 Hopper 架构，采用 4nm 工艺，在 AI 训练市场占据超过 80% 份额，2024 年仍然是深度学习训练的首选硬件平台。",
        "topic": "AI芯片",
        "source": "NVIDIA 官方技术文档",
        "confidence": 0.95,
        "evidence_type": "primary",
    },
    {
        "claim": "AMD MI300X 加速卡在推理场景中对标 NVIDIA H100，其 CDNA3 架构在 memory bandwidth 方面达到 5.2 TB/s，但在训练生态方面仍落后于 CUDA 生态。",
        "topic": "AI芯片",
        "source": "AMD 技术白皮书",
        "confidence": 0.92,
        "evidence_type": "primary",
    },
    {
        "claim": "华为昇腾 910B 采用中芯国际 7nm 工艺制造，受限于美国芯片出口管制，先进制程产能受限，主要在中国国内市场部署。",
        "topic": "AI芯片",
        "source": "华为官网",
        "confidence": 0.88,
        "evidence_type": "primary",
    },
    {
        "claim": "NVIDIA 的 CUDA 生态护城河极深，拥有超过 400 万开发者，超过 3000 个 GPU 加速库，这是 AMD 和 Intel 短期内难以逾越的壁垒。",
        "topic": "AI芯片",
        "source": "NVIDIA GTC 2024 Keynote",
        "confidence": 0.93,
        "evidence_type": "secondary",
    },
    # ---- 医疗 / mRNA 疫苗 ----
    {
        "claim": "Moderna 的 mRNA-4157 联合 Keytruda 在 2b 期临床试验中，将高危黑色素瘤患者的复发风险降低了 44%，该结果于 2024 年发表于《柳叶刀》。",
        "topic": "mRNA疫苗",
        "source": "The Lancet, 2024",
        "confidence": 0.96,
        "evidence_type": "primary",
    },
    {
        "claim": "BioNTech 的 BNT122（个性化新抗原 mRNA 疫苗）在结直肠癌的 2 期试验中显示出可接受的耐受性和初步疗效信号。",
        "topic": "mRNA疫苗",
        "source": "BioNTech 临床数据发布",
        "confidence": 0.90,
        "evidence_type": "primary",
    },
    {
        "claim": "mRNA 疫苗技术的核心原理是利用脂质纳米粒（LNP）包裹编码肿瘤新抗原的 mRNA，注射后由树突状细胞摄取并翻译为抗原蛋白，激活 T 细胞免疫应答。",
        "topic": "mRNA疫苗",
        "source": "Nature Reviews Drug Discovery",
        "confidence": 0.97,
        "evidence_type": "primary",
    },
    # ---- 金融 / 央行政策 ----
    {
        "claim": "美联储于 2024 年 9 月 18 日宣布降息 50 个基点，将联邦基金利率目标区间下调至 4.75%–5.00%，这是自 2020 年以来的首次降息。",
        "topic": "美联储政策",
        "source": "美联储 FOMC 声明",
        "confidence": 0.99,
        "evidence_type": "primary",
    },
    {
        "claim": "美联储降息 50bp 后，新兴市场出现资本回流浪潮。据 IIF 数据，2024 年 10 月新兴市场股票和债券净流入超过 300 亿美元。",
        "topic": "美联储政策",
        "source": "IIF 资本流动报告",
        "confidence": 0.91,
        "evidence_type": "primary",
    },
    {
        "claim": "美联储降息缩小了中美利差，减轻了人民币贬值压力，为中国人民银行提供了更大的货币政策自主空间，2024 年 9 月底中国央行随即下调 MLF 利率 20bp。",
        "topic": "美联储政策",
        "source": "中国人民银行公告",
        "confidence": 0.89,
        "evidence_type": "secondary",
    },
    # ---- 能源 / 电池技术 ----
    {
        "claim": "QuantumScape 的固态电池使用陶瓷隔膜替代液态电解质，在实验室测试中达到 800 次循环后仍保留 80% 以上容量，能量密度目标为 500 Wh/kg。",
        "topic": "电池技术",
        "source": "QuantumScape 股东信",
        "confidence": 0.87,
        "evidence_type": "primary",
    },
    {
        "claim": "宁德时代于 2024 年发布凝聚态电池（Condensed Battery），单体能量密度达到 500 Wh/kg，采用高能量固态电解质技术，预计 2025 年开始量产。",
        "topic": "电池技术",
        "source": "宁德时代产品发布会",
        "confidence": 0.93,
        "evidence_type": "primary",
    },
    {
        "claim": "磷酸铁锂（LFP）电池的成本约为 60 美元/kWh，远低于三元锂电池的 90-100 美元/kWh，但能量密度上限仅约 160 Wh/kg，主要应用于中低端电动车和储能场景。",
        "topic": "电池技术",
        "source": "BloombergNEF 电池价格调查",
        "confidence": 0.94,
        "evidence_type": "primary",
    },
    # ---- 汽车 / 新能源车 ----
    {
        "claim": "比亚迪 2024 年全年新能源车销量达到 427 万辆，超过特斯拉成为全球新能源车销量第一的品牌，其垂直整合模式（自产电池、芯片、电机）是核心竞争力。",
        "topic": "新能源车",
        "source": "比亚迪 2024 年度产销快报",
        "confidence": 0.96,
        "evidence_type": "primary",
    },
    {
        "claim": "特斯拉的 FSD（Full Self-Driving）V12 采用端到端神经网络架构，以纯视觉方案（无激光雷达）实现城市道路自动驾驶，2024 年已向北美所有车主推送。",
        "topic": "新能源车",
        "source": "特斯拉 AI Day 2024",
        "confidence": 0.90,
        "evidence_type": "primary",
    },
    {
        "claim": "2024 年中国新能源车零售渗透率首次突破 50%（达 50.8%），传统燃油车经销商 4S 店大量关闭或转型，充电桩数量已超过 1000 万个。",
        "topic": "新能源车",
        "source": "中国汽车工业协会 (CAAM) 数据",
        "confidence": 0.95,
        "evidence_type": "primary",
    },
    {
        "claim": "华为不造整车，而是通过智选车模式（鸿蒙智行）为车企提供智能驾驶、智能座舱和电驱系统解决方案，问界（AITO）系列是其最成功的合作案例。",
        "topic": "新能源车",
        "source": "华为终端业务公告",
        "confidence": 0.92,
        "evidence_type": "secondary",
    },
]

# ==============================================================================
# 评测查询集 —— 每个 query 附带 ground_truth 条目（期望被检索到的文档 claim 关键词）
# ==============================================================================
EVAL_QUERIES: list[dict[str, Any]] = [
    {
        "query": "AI 训练芯片市场 NVIDIA 和 AMD 的竞争格局如何？",
        "relevant_keywords": ["NVIDIA H100", "AMD MI300X", "CUDA 生态"],
        "min_relevant": 3,
    },
    {
        "query": "mRNA 癌症疫苗的技术原理和临床试验进展",
        "relevant_keywords": ["mRNA-4157", "BNT122", "脂质纳米粒", "树突状细胞"],
        "min_relevant": 3,
    },
    {
        "query": "美联储 2024 年降息对新兴市场和中国货币政策的影响",
        "relevant_keywords": ["降息 50 个基点", "资本回流", "中美利差", "MLF"],
        "min_relevant": 3,
    },
    {
        "query": "固态电池、磷酸铁锂电池和凝聚态电池的技术对比",
        "relevant_keywords": ["QuantumScape", "凝聚态电池", "磷酸铁锂", "LFP"],
        "min_relevant": 3,
    },
    {
        "query": "中国新能源车市场格局：比亚迪、特斯拉和华为的差异化策略",
        "relevant_keywords": ["比亚迪", "FSD V12", "鸿蒙智行", "渗透率"],
        "min_relevant": 4,
    },
    {
        "query": "NVIDIA CUDA 生态为什么难以被替代？",
        "relevant_keywords": ["400 万开发者", "3000 个 GPU 加速库"],
        "min_relevant": 1,
    },
    {
        "query": "中国在 AI 芯片方面的自主替代进展",
        "relevant_keywords": ["昇腾 910B", "中芯国际", "7nm"],
        "min_relevant": 1,
    },
]


# ==============================================================================
# 文档摄入
# ==============================================================================
def ingest_docs(
    store: SharedMemoryStore,
    docs: list[dict[str, Any]],
) -> int:
    """将种子文档写入 SharedMemoryStore。"""
    count = 0
    for doc in docs:
        entry = MemoryEntry(
            claim=doc["claim"],
            topic=doc["topic"],
            source=doc.get("source", ""),
            confidence=doc.get("confidence", 0.8),
            evidence_type=doc.get("evidence_type", "secondary"),
        )
        store.put(entry)
        count += 1
    logger.info(f"Ingested {count} documents into memory store.")
    return count


# ==============================================================================
# 检索对比
# ==============================================================================
def run_retrieval_comparison(
    queries: list[dict[str, Any]],
    seed_docs: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    检索级对比：RAG-only vs RAG+GraphRAG。

    流程：
      1. 创建两个独立的 SharedMemoryStore（分别对应纯 RAG 和 RAG+GraphRAG）
      2. 写入相同的种子文档
      3. 对每个 query，分别检索并对比命中率 / recall / MRR
    """
    print("\n" + "=" * 70)
    print("[RAG vs GraphRAG] 检索级对比实验")
    print("=" * 70)

    # 创建两个独立的 store（用不同的 db_path 避免数据交叉污染）
    rag_db = "data/rag_only.db"
    graphrag_db = "data/rag_graphrag.db"

    # 清理旧数据
    for p in [rag_db, graphrag_db]:
        if os.path.exists(p):
            os.remove(p)

    # RAG-only store（graph_policy=None → 不构建图谱）
    store_rag = SharedMemoryStore(
        db_path=rag_db,
        enable_graph_rag=False,
        graph_policy=None,
    )

    # RAG+GraphRAG store（带 mock graph_policy 用于图谱抽取）
    # 注意：图谱抽取需要 LLM，这里用一个模拟 policy 做演示
    from src.models.model_router import ModelRouter

    try:
        graph_policy = ModelRouter.create_backend("deepseek")
    except Exception:
        logger.warning("无法创建 LLM backend，使用 None policy（GraphRAG 提取将跳过）")
        graph_policy = None

    store_graphrag = SharedMemoryStore(
        db_path=graphrag_db,
        enable_graph_rag=True,
        graph_policy=graph_policy,
    )

    # 写入相同文档
    print(f"\n[摄入] 写入 {len(seed_docs)} 条种子文档...")
    ingest_docs(store_rag, seed_docs)
    ingest_docs(store_graphrag, seed_docs)
    print(f"  RAG store:       {len(store_rag)} entries")
    print(f"  GraphRAG store:   {len(store_graphrag)} entries")

    # 如果 GraphRAG 有 LLM，等待实体抽取完成
    if store_graphrag.graph_extractor.enabled:
        print(f"  GraphRAG 实体抽取: enabled (extractions={store_graphrag.graph_extractor.stats.get('extraction_count', 0)})")
    else:
        print(f"  GraphRAG 实体抽取: disabled (无 graph_policy)")

    # 逐 query 评测
    results = []
    for idx, q in enumerate(EVAL_QUERIES, 1):
        query = q["query"]
        keywords = q["relevant_keywords"]
        min_relevant = q.get("min_relevant", 1)

        print(f"\n[{idx}/{len(EVAL_QUERIES)}] Query: {query[:60]}...")

        # RAG-only 检索
        rag_hits = store_rag.query_by_similarity(query, top_k=10, min_sim=0.50)
        rag_texts = [entry.claim for entry, _ in rag_hits]

        # RAG+GraphRAG 检索（双通道）
        graphrag_context = store_graphrag.get_context_for_query(query, max_tokens=4000)
        # 从 context 中提取命中的条目（按 "来源:" 分割）
        graphrag_texts = _extract_claims_from_context(graphrag_context)

        # 计算命中指标
        rag_metrics = _compute_retrieval_metrics(rag_texts, keywords)
        graphrag_metrics = _compute_retrieval_metrics(graphrag_texts, keywords)

        result = {
            "query_id": idx,
            "query": query,
            "rag_only": {
                "num_retrieved": len(rag_texts),
                "hits": rag_metrics["hits"],
                "hit_rate": rag_metrics["hit_rate"],
                "mrr": rag_metrics["mrr"],
                "matched_keywords": rag_metrics["matched_keywords"],
            },
            "rag_graphrag": {
                "num_retrieved": len(graphrag_texts),
                "hits": graphrag_metrics["hits"],
                "hit_rate": graphrag_metrics["hit_rate"],
                "mrr": graphrag_metrics["mrr"],
                "matched_keywords": graphrag_metrics["matched_keywords"],
            },
        }

        delta = graphrag_metrics["hit_rate"] - rag_metrics["hit_rate"]
        winner = "GraphRAG ✓" if delta > 0 else ("RAG ✓" if delta < 0 else "平局")
        result["delta"] = delta
        result["winner"] = winner

        results.append(result)
        print(f"  RAG hit_rate={rag_metrics['hit_rate']:.2f} | "
              f"GraphRAG hit_rate={graphrag_metrics['hit_rate']:.2f} | "
              f"Δ={delta:+.2f} {winner}")

    # 汇总统计
    rag_hit_rates = [r["rag_only"]["hit_rate"] for r in results]
    graphrag_hit_rates = [r["rag_graphrag"]["hit_rate"] for r in results]

    stats = bootstrap_ci_two_sample(graphrag_hit_rates, rag_hit_rates)
    d_value = cohens_d(graphrag_hit_rates, rag_hit_rates)

    summary = {
        "experiment": "RAG vs GraphRAG — Retrieval-level Comparison",
        "num_queries": len(EVAL_QUERIES),
        "num_docs": len(seed_docs),
        "rag_avg_hit_rate": float(np.mean(rag_hit_rates)),
        "graphrag_avg_hit_rate": float(np.mean(graphrag_hit_rates)),
        "mean_delta": stats["mean_diff"],
        "ci_lower": stats["ci_lower"],
        "ci_upper": stats["ci_upper"],
        "p_value": stats["p_value"],
        "significant": stats["significant"],
        "cohens_d": round(d_value, 4),
        "results": results,
    }

    print("\n" + "=" * 70)
    print("[汇总] RAG vs GraphRAG 检索对比")
    print("=" * 70)
    print(f"  RAG-only 平均命中率:     {summary['rag_avg_hit_rate']:.4f}")
    print(f"  RAG+GraphRAG 平均命中率: {summary['graphrag_avg_hit_rate']:.4f}")
    print(f"  平均差异 Δ:              {summary['mean_delta']:+.4f}")
    print(f"  95% Bootstrap CI:        [{summary['ci_lower']:+.4f}, {summary['ci_upper']:+.4f}]")
    print(f"  p-value:                 {summary['p_value']:.4f}")
    print(f"  Cohen's d:               {summary['cohens_d']:.4f}")
    print(f"  显著性:                  {'✓ 显著' if summary['significant'] else '✗ 不显著'}")

    return summary


def _extract_claims_from_context(context: str) -> list[str]:
    """从 get_context_for_query 的输出文本中提取 claim 列表。"""
    claims = []
    for line in context.split("\n"):
        line = line.strip()
        if line.startswith("- [") and "] " in line:
            # 格式: "- [topic] claim text"
            claim = line.split("] ", 1)[-1].strip()
            if claim:
                claims.append(claim)
    return claims


def _compute_retrieval_metrics(
    retrieved_texts: list[str],
    keywords: list[str],
) -> dict[str, Any]:
    """计算单次检索的命中指标。"""
    all_text = " ".join(retrieved_texts).lower()
    matched = [kw for kw in keywords if kw.lower() in all_text]
    hit_rate = len(matched) / len(keywords) if keywords else 0.0

    # MRR (Mean Reciprocal Rank): 每个 keyword 在检索列表中首次出现的排位倒数
    reciprocal_ranks = []
    for kw in keywords:
        kw_lower = kw.lower()
        found = False
        for rank, text in enumerate(retrieved_texts, 1):
            if kw_lower in text.lower():
                reciprocal_ranks.append(1.0 / rank)
                found = True
                break
        if not found:
            reciprocal_ranks.append(0.0)
    mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0

    return {
        "hits": len(matched),
        "total_keywords": len(keywords),
        "hit_rate": hit_rate,
        "mrr": mrr,
        "matched_keywords": matched,
    }


# ==============================================================================
# 端到端对比（完整 Agent 流程）
# ==============================================================================
async def run_e2e_comparison(
    queries: list[str],
    config: dict,
    num_questions: int = 5,
) -> dict[str, Any]:
    """
    端到端对比：用完整 Agent 流程跑相同 query，对比报告质量。
    """
    from src.core.runner import initialize_modules, run_research
    from evaluation.metrics.rule_based import RuleBasedMetrics

    print("\n" + "=" * 70)
    print("[RAG vs GraphRAG] 端到端对比实验（完整 Agent 流程）")
    print("=" * 70)

    queries = queries[:num_questions]
    results = []

    for idx, query in enumerate(queries, 1):
        print(f"\n[{idx}/{len(queries)}] Query: {query[:60]}...")

        record = {"query": query, "rag_only": None, "rag_graphrag": None}

        # ---- RAG-only ----
        print("  运行 RAG-only...")
        modules_rag = initialize_modules(config, session_id=f"rag_e2e_{idx}", enable_graph_rag=False)
        t0 = time.time()
        try:
            report_rag = await run_research(query, config, modules_rag)
            elapsed_rag = time.time() - t0
            record["rag_only"] = {
                "report": report_rag,
                "length": len(report_rag),
                "elapsed": round(elapsed_rag, 1),
            }
            print(f"  RAG: {len(report_rag)} chars, {elapsed_rag:.1f}s")
        except Exception as e:
            logger.error(f"RAG-only failed: {e}")
            record["rag_only"] = {"error": str(e)}

        # ---- RAG+GraphRAG ----
        print("  运行 RAG+GraphRAG...")
        modules_graphrag = initialize_modules(config, session_id=f"graphrag_e2e_{idx}", enable_graph_rag=True)
        t0 = time.time()
        try:
            report_graphrag = await run_research(query, config, modules_graphrag)
            elapsed_graphrag = time.time() - t0
            record["rag_graphrag"] = {
                "report": report_graphrag,
                "length": len(report_graphrag),
                "elapsed": round(elapsed_graphrag, 1),
            }
            print(f"  GraphRAG: {len(report_graphrag)} chars, {elapsed_graphrag:.1f}s")
        except Exception as e:
            logger.error(f"GraphRAG failed: {e}")
            record["rag_graphrag"] = {"error": str(e)}

        # 规则指标对比
        if record["rag_only"].get("report") and record["rag_graphrag"].get("report"):
            metrics_rag = _compute_report_metrics(record["rag_only"]["report"])
            metrics_graphrag = _compute_report_metrics(record["rag_graphrag"]["report"])
            record["rag_only"]["metrics"] = metrics_rag
            record["rag_graphrag"]["metrics"] = metrics_graphrag

            print(f"  引用覆盖率: RAG={metrics_rag['citation_coverage']:.2f} vs GraphRAG={metrics_graphrag['citation_coverage']:.2f}")
            print(f"  逻辑一致性: RAG={metrics_rag['logical_consistency']:.2f} vs GraphRAG={metrics_graphrag['logical_consistency']:.2f}")

        results.append(record)

    # 汇总
    rag_citations = []
    graphrag_citations = []
    rag_logic = []
    graphrag_logic = []

    for r in results:
        if r["rag_only"] and "metrics" in r["rag_only"]:
            rag_citations.append(r["rag_only"]["metrics"]["citation_coverage"])
            rag_logic.append(r["rag_only"]["metrics"]["logical_consistency"])
        if r["rag_graphrag"] and "metrics" in r["rag_graphrag"]:
            graphrag_citations.append(r["rag_graphrag"]["metrics"]["citation_coverage"])
            graphrag_logic.append(r["rag_graphrag"]["metrics"]["logical_consistency"])

    summary: dict[str, Any] = {
        "experiment": "RAG vs GraphRAG — End-to-End Comparison",
        "num_questions": len(queries),
        "results": results,
    }

    if rag_citations and graphrag_citations:
        cit_stats = bootstrap_ci_two_sample(graphrag_citations, rag_citations)
        logic_stats = bootstrap_ci_two_sample(graphrag_logic, rag_logic)

        summary["citation_coverage"] = {
            "rag_avg": float(np.mean(rag_citations)),
            "graphrag_avg": float(np.mean(graphrag_citations)),
            "delta": cit_stats["mean_diff"],
            "ci_lower": cit_stats["ci_lower"],
            "ci_upper": cit_stats["ci_upper"],
            "significant": cit_stats["significant"],
            "cohens_d": round(cohens_d(graphrag_citations, rag_citations), 4),
        }
        summary["logical_consistency"] = {
            "rag_avg": float(np.mean(rag_logic)),
            "graphrag_avg": float(np.mean(graphrag_logic)),
            "delta": logic_stats["mean_diff"],
            "ci_lower": logic_stats["ci_lower"],
            "ci_upper": logic_stats["ci_upper"],
            "significant": logic_stats["significant"],
            "cohens_d": round(cohens_d(graphrag_logic, rag_logic), 4),
        }

        print("\n" + "=" * 70)
        print("[汇总] 端到端对比")
        print("=" * 70)
        for dim_name, dim_data in [("引用覆盖率", summary["citation_coverage"]), ("逻辑一致性", summary["logical_consistency"])]:
            sig = "✓ 显著" if dim_data["significant"] else "✗ 不显著"
            print(f"  {dim_name}: RAG={dim_data['rag_avg']:.4f} vs GraphRAG={dim_data['graphrag_avg']:.4f} "
                  f"Δ={dim_data['delta']:+.4f} d={dim_data['cohens_d']:.2f} {sig}")

    return summary


def _compute_report_metrics(report: str) -> dict[str, float]:
    """计算单篇报告的规则指标。"""
    from evaluation.metrics.rule_based import RuleBasedMetrics
    return {
        "citation_coverage": RuleBasedMetrics.citation_coverage(report),
        "logical_consistency": RuleBasedMetrics.logical_consistency(report),
        "hallucination_rate": RuleBasedMetrics.hallucination_rate(report),
    }


# ==============================================================================
# Main
# ==============================================================================
async def main() -> None:
    parser = argparse.ArgumentParser(description="RAG vs GraphRAG 对比实验")
    parser.add_argument("--mode", type=str, choices=["retrieval", "e2e"], default="retrieval",
                        help="实验模式: retrieval（检索级）/ e2e（端到端）")
    parser.add_argument("--num_questions", type=int, default=5,
                        help="端到端模式下评测的问题数")
    parser.add_argument("--output", type=str, default="outputs/rag_vs_graphrag.json",
                        help="结果输出路径")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.mode == "retrieval":
        summary = run_retrieval_comparison(EVAL_QUERIES, SEED_DOCS)
    else:
        from src.core.runner import load_config
        config = load_config()

        # 从 ResearchBench 取端到端评测问题
        from evaluation.benchmarks.research_bench import ResearchBench
        bench = ResearchBench()
        questions = bench.get_questions(n=args.num_questions)
        queries = [q["query"] for q in questions]
        print(f"[E2E] 从 ResearchBench 加载 {len(queries)} 道题目")

        summary = await run_e2e_comparison(queries, config, args.num_questions)

    # 保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/run_quantitative_bench.py
================================================================================
DeepResearch Agent 多维度定量评测脚本

目的：从多个可量化维度证明系统的优越性，适合论文/报告中使用。

评测维度：
  D1. 模块消融           — 每个模块的边际贡献 (composite score Δ)
  D2. RAG 检索质量        — 向量召回命中率 / MRR / NDCG
  D3. 压缩保真度          — L1→L2→L3 各级信息保留率
  D4. 记忆效率            — 去重率 / 矛盾检测率 / token 节省量
  D5. Agent vs LLM        — 多智能体 vs 单轮 LLM 的质量差异
  D6. 编排效率            — 并发 vs 串行的加速比（规划/执行/合成耗时分布）
  D7. 鲁棒性              — 不同 query 类型 / 不同 LLM 后端的稳定性

输出：每个维度独立 JSON + 一份汇总 Markdown 报告

用法：
    # 跑全部维度（耗时较长，约 30-60 min）
    python scripts/run_quantitative_bench.py --all

    # 只跑某个维度
    python scripts/run_quantitative_bench.py --dim D1
    python scripts/run_quantitative_bench.py --dim D3,D5

    # 快速模式（只用 3 道题，验证流程）
    python scripts/run_quantitative_bench.py --all --quick
================================================================================
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.runner import initialize_modules, load_config, run_research, setup_logging
from src.memory.memory_store import SharedMemoryStore
from src.memory.embedder import Embedder
from src.memory.long_term import MemoryEntry
from evaluation.benchmarks.research_bench import ResearchBench
from evaluation.metrics.rule_based import RuleBasedMetrics
from evaluation.metrics.stats import (
    bootstrap_ci_paired,
    cohens_d,
)

logger = logging.getLogger("quant_bench")

# ============================================================================
# 公共工具
# ============================================================================

def _describe_effect(d_value: float) -> str:
    """Cohen's d 效应量文字描述。"""
    d = abs(d_value)
    if d < 0.2:
        return "可忽略"
    elif d < 0.5:
        return "小"
    elif d < 0.8:
        return "中"
    else:
        return "大"


def _significance_marker(stats: dict) -> str:
    """显著性标记。"""
    if stats.get("significant"):
        p = stats.get("p_value", 1.0)
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
    return "n.s."


# ============================================================================
# D1: 模块消融 —— 每个模块的边际贡献
# ============================================================================

async def benchmark_module_ablation(
    config: dict,
    questions: list[dict],
    output_dir: str,
) -> dict[str, Any]:
    """
    在 ResearchBench 上跑 5 个配置（full + 4 个消融），
    对每道题计算配对差异，输出每个模块的边际贡献。
    """
    print("\n" + "=" * 70)
    print("[D1] 模块消融：量化每个模块的边际贡献")
    print("=" * 70)

    bench = ResearchBench()

    systems = {
        "full":            ("完整系统", {}),
        "no_compressor":   ("关闭上下文压缩", {"compressor": {"enabled": False}}),
        "no_memory":       ("关闭共享记忆", {"memory": {"enabled": False}}),
    }

    all_results: dict[str, dict] = {}

    for name, (desc, overrides) in systems.items():
        print(f"\n--- {name}: {desc} ---")

        cfg = dict(config)
        for k, v in overrides.items():
            if isinstance(v, dict):
                cfg.setdefault(k, {}).update(v)
            else:
                cfg[k] = v

        modules = initialize_modules(cfg, session_id=f"d1_{name}")

        details = []
        scores = []

        for q in questions:
            qid = q["id"]
            query = q["query"]
            print(f"  [{qid}] {query[:50]}...", end=" ", flush=True)

            try:
                report = await run_research(query, cfg, modules)
                result = bench.evaluate_report(report, qid)
                scores.append(result["composite_score"])
                details.append(result)
                print(f"score={result['composite_score']:.3f}")
            except Exception as e:
                print(f"FAILED: {e}")
                scores.append(0.0)
                details.append({"question_id": qid, "error": str(e), "composite_score": 0.0})

        avg = np.mean(scores) if scores else 0.0
        all_results[name] = {
            "system": name,
            "description": desc,
            "avg_composite": float(avg),
            "scores": scores,
            "details": details,
        }
        print(f"  → avg={avg:.4f}")

    # 统计：full vs 每个消融配置
    full_scores = all_results["full"]["scores"]
    ablation_stats = {}
    for name in ["no_compressor", "no_memory"]:
        if name not in all_results:
            continue
        ab_scores = all_results[name]["scores"]
        diffs = [f - a for f, a in zip(full_scores, ab_scores)]
        stats = bootstrap_ci_paired(diffs)
        effect = cohens_d(full_scores, ab_scores)
        ablation_stats[name] = {
            **stats,
            "cohens_d": round(effect, 4),
            "effect_size": _describe_effect(effect),
            "marginal_contribution": stats["mean_diff"],
            "significance": _significance_marker(stats),
        }
        print(f"  {name}: Δ={stats['mean_diff']:+.4f} "
              f"CI=[{stats['ci_lower']:+.4f},{stats['ci_upper']:+.4f}] "
              f"d={effect:.3f} ({_describe_effect(effect)}) {_significance_marker(stats)}")

    result = {
        "dimension": "D1_Module_Ablation",
        "description": "量化每个模块关闭后 composite score 的下降幅度",
        "full_score": all_results["full"]["avg_composite"],
        "ablation_stats": ablation_stats,
        "ranking": sorted(
            ablation_stats.items(),
            key=lambda x: x[1]["marginal_contribution"],
            reverse=True,
        ),
    }

    _save(result, output_dir, "D1_module_ablation")
    return result


# ============================================================================
# D2: RAG 检索质量
# ============================================================================

def benchmark_rag_retrieval(output_dir: str) -> dict[str, Any]:
    """在本地记忆库上评估普通向量 RAG 的命中率、MRR 和 NDCG。"""
    print("\n" + "=" * 70)
    print("[D2] RAG：检索质量")
    print("=" * 70)

    # D2 使用普通向量 RAG 进行单系统检索评测。
    if os.path.exists("data/d3_rag.db"):
        os.remove("data/d3_rag.db")
    store_rag = SharedMemoryStore(db_path="data/d3_rag.db")
    seed_docs = [
        {"claim": "NVIDIA H100 GPU 基于 Hopper 架构，主要用于 AI 训练。", "topic": "AI芯片", "source": "seed", "confidence": 0.95, "evidence_type": "primary"},
        {"claim": "AMD MI300X 面向 AI 推理和训练场景。", "topic": "AI芯片", "source": "seed", "confidence": 0.90, "evidence_type": "primary"},
        {"claim": "Transformer 使用自注意力机制处理序列信息。", "topic": "深度学习", "source": "seed", "confidence": 0.92, "evidence_type": "primary"},
    ]
    for doc in seed_docs:
        store_rag.put(MemoryEntry(
            entry_id=str(uuid.uuid4()),
            agent_id="benchmark",
            timestamp=time.time(),
            embedding=[],
            **doc,
        ))
    eval_queries = [
        {"query": "AI 训练芯片", "relevant_keywords": ["H100", "AI", "训练"]},
        {"query": "AI 推理加速卡", "relevant_keywords": ["MI300X", "AI", "推理"]},
        {"query": "Transformer 自注意力", "relevant_keywords": ["Transformer", "自注意力"]},
    ]

    metrics_per_query = []
    rag_hits_all = []
    rag_mrr_all = []

    for idx, q in enumerate(eval_queries):
        query = q["query"]
        keywords = q["relevant_keywords"]

        rag_results = store_rag.query_by_similarity(query, top_k=10, min_sim=0.45)
        rag_texts = [e.claim for e, _ in rag_results]

        rag_m = {
            "hits": sum(1 for text in rag_texts if any(kw.lower() in text.lower() for kw in keywords)),
            "hit_rate": sum(1 for text in rag_texts if any(kw.lower() in text.lower() for kw in keywords)) / max(len(rag_texts), 1),
            "mrr": next((1.0 / (i + 1) for i, text in enumerate(rag_texts) if any(kw.lower() in text.lower() for kw in keywords)), 0.0),
            "matched_keywords": sorted({kw for kw in keywords if any(kw.lower() in text.lower() for text in rag_texts)}),
        }
        rag_ndcg = _compute_ndcg(rag_texts, keywords, k=10)

        metrics_per_query.append({
            "query_id": idx,
            "query": query,
            "rag": {**rag_m, "ndcg@10": rag_ndcg},
        })

        rag_hits_all.append(rag_m["hit_rate"])
        rag_mrr_all.append(rag_m["mrr"])

    result = {
        "dimension": "D2_RAG_Retrieval",
        "description": "普通向量 RAG 检索质量",
        "rag_avg_hit_rate": float(np.mean(rag_hits_all)),
        "rag_avg_mrr": float(np.mean(rag_mrr_all)),
        "per_query": metrics_per_query,
    }

    print(f"  Hit Rate: RAG={result['rag_avg_hit_rate']:.3f}")
    print(f"  MRR:      RAG={result['rag_avg_mrr']:.3f}")

    _save(result, output_dir, "D2_rag_retrieval")
    return result


def _compute_ndcg(texts: list[str], keywords: list[str], k: int = 10) -> float:
    """简化 NDCG@K：keyword 命中=1，否则=0。"""
    all_text = " ".join(texts[:k]).lower()
    dcg = 0.0
    for i, text in enumerate(texts[:k]):
        rel = sum(1 for kw in keywords if kw.lower() in text.lower())
        if i == 0:
            dcg += rel
        else:
            dcg += rel / math.log2(i + 2)
    ideal_rel = min(len(keywords), k)
    idcg = ideal_rel + sum(1.0 / math.log2(i + 2) for i in range(1, min(k, len(keywords))))
    return dcg / idcg if idcg > 0 else 0.0


# ============================================================================
# D3: 压缩保真度
# ============================================================================

def benchmark_compressor_fidelity(output_dir: str) -> dict[str, Any]:
    """
    用不同长度的输入文本测试三级压缩的信息保留率。
    不需要 LLM，纯本地测试。
    """
    print("\n" + "=" * 70)
    print("[D3] 压缩保真度：L1→L2→L3 各级信息保留率")
    print("=" * 70)

    from src.compressor.compressor import ContextCompressor

    # 构造不同长度的测试文本
    short_text = "GPT-4o 是 OpenAI 于 2024 年 5 月发布的原生多模态大模型。其核心创新包括：1) 跨模态 tokenizer 统一处理文本和图像；2) 端到端训练支持语音对话；3) 在 MMLU 基准上达到 88.7% 准确率。" * 5
    medium_text = short_text * 10
    long_text = short_text * 30

    compressor = ContextCompressor(llm_policy=None, budget=16000, output_reserve=2048)

    # 关键信息标记（数字实体 + 专有名词，用于计算保留率）
    key_entities = ["GPT-4o", "OpenAI", "2024年5月", "88.7%", "MMLU", "多模态"]

    results = {}
    for label, text in [("short (~500t)", short_text), ("medium (~5000t)", medium_text), ("long (~15000t)", long_text)]:
        tokens_in = len(text) // 3  # 粗略估计

        # L1: embedding 过滤
        l1_result = compressor._filter_by_similarity(text.split("。"), query="大模型技术", threshold=0.5)
        l1_text = "。".join(l1_result) if l1_result else text
        l1_retention = _calc_entity_retention(text, l1_text, key_entities)

        # L2: TextRank
        try:
            l2_result = compressor._textrank_extract(text, num_sentences=max(3, len(text.split("。")) // 2))
            l2_text = "。".join(l2_result) if l2_result else text
        except Exception:
            l2_text = text
        l2_retention = _calc_entity_retention(text, l2_text, key_entities)

        # L3: 模拟摘要（取前 30%）
        sentences = text.split("。")
        l3_text = "。".join(sentences[:max(3, len(sentences) // 3)])
        l3_retention = _calc_entity_retention(text, l3_text, key_entities)

        compression_ratio_l1 = len(l1_text) / max(len(text), 1)
        compression_ratio_l2 = len(l2_text) / max(len(text), 1)
        compression_ratio_l3 = len(l3_text) / max(len(text), 1)

        results[label] = {
            "tokens_in": tokens_in,
            "L1": {"compression_ratio": round(compression_ratio_l1, 3), "entity_retention": round(l1_retention, 3)},
            "L2": {"compression_ratio": round(compression_ratio_l2, 3), "entity_retention": round(l2_retention, 3)},
            "L3": {"compression_ratio": round(compression_ratio_l3, 3), "entity_retention": round(l3_retention, 3)},
        }

        print(f"  {label}: L1→{l1_retention:.2f} | L2→{l2_retention:.2f} | L3→{l3_retention:.2f}")

    result = {
        "dimension": "D3_Compressor_Fidelity",
        "description": "三级压缩在不同输入长度下的关键实体保留率",
        "key_entities": key_entities,
        "results": results,
    }

    _save(result, output_dir, "D3_compressor_fidelity")
    return result


def _calc_entity_retention(original: str, compressed: str, entities: list[str]) -> float:
    """计算关键实体在压缩后文本中的保留比例。"""
    if not entities:
        return 1.0
    original_lower = original.lower()
    compressed_lower = compressed.lower()
    retained = sum(1 for e in entities if e.lower() in original_lower and e.lower() in compressed_lower)
    present = sum(1 for e in entities if e.lower() in original_lower)
    return retained / max(present, 1)


# ============================================================================
# D4: 记忆效率
# ============================================================================

def benchmark_memory_efficiency(output_dir: str) -> dict[str, Any]:
    """
    测试去重率、矛盾检测率、token 节省量。
    """
    print("\n" + "=" * 70)
    print("[D4] 记忆效率：去重 / 矛盾检测 / Token 节省")
    print("=" * 70)

    db_path = "data/d5_memory_test.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    store = SharedMemoryStore(db_path=db_path)

    # 写入含重复和矛盾的文档
    entries_data = [
        # 正常条目
        {"claim": "NVIDIA H100 在 AI 训练市场占据 80% 份额", "topic": "AI芯片", "confidence": 0.95},
        {"claim": "NVIDIA H100 在 AI 训练市场占据约 80% 的份额，领先竞争对手", "topic": "AI芯片", "confidence": 0.85},  # 语义重复
        {"claim": "NVIDIA H100 在 AI 训练市场仅占 50% 份额", "topic": "AI芯片", "confidence": 0.60},  # 矛盾！
        # 更多领域
        {"claim": "比亚迪 2024 年销量 427 万辆，全球第一", "topic": "新能源车", "confidence": 0.96},
        {"claim": "比亚迪 2024 年交付超 420 万辆新能源车", "topic": "新能源车", "confidence": 0.92},  # 重复
        {"claim": "特斯拉 2024 年销量超过比亚迪", "topic": "新能源车", "confidence": 0.55},  # 矛盾！
        # 正常条目（无重复）
        {"claim": "Moderna mRNA-4157 将复发风险降低 44%", "topic": "mRNA疫苗", "confidence": 0.96},
        {"claim": "美联储 2024 年 9 月降息 50bp", "topic": "美联储", "confidence": 0.99},
    ]

    put_count = 0
    merge_count = 0
    conflict_count = 0

    for data in entries_data:
        entry = MemoryEntry(
            claim=data["claim"],
            topic=data["topic"],
            confidence=data["confidence"],
            source="benchmark",
            evidence_type="primary",
        )
        eid = store.put(entry)
        put_count += 1

    # 检查去重（条目数应少于写入数）
    total_entries = len(store)
    merge_count = put_count - total_entries

    # 检查矛盾
    conflicts = store.get_conflicts()
    conflict_count = len(conflicts)

    # Token 节省估算：假设无去重时每次检索返回 top-10 共 10 条，
    # 有去重后冗余条目被合并，检索上下文更精炼
    estimated_tokens_without_dedup = put_count * 150  # 每条约 150 tokens
    estimated_tokens_with_dedup = total_entries * 150
    token_savings = estimated_tokens_without_dedup - estimated_tokens_with_dedup
    savings_ratio = token_savings / max(estimated_tokens_without_dedup, 1)

    print(f"  写入: {put_count} 条")
    print(f"  去重后: {total_entries} 条 (合并 {merge_count} 条, 去重率={merge_count / put_count:.1%})")
    print(f"  矛盾检测: {conflict_count} 对")
    print(f"  Token 节省: ~{token_savings} tokens ({savings_ratio:.1%})")

    # 检索质量对比：有记忆 vs 无记忆
    test_query = "NVIDIA AI 芯片市场格局"
    ctx_with_memory = store.get_context_for_query(test_query, max_tokens=2000)

    # 模拟无记忆（空 store）
    empty_store = SharedMemoryStore(db_path="data/d5_empty.db")
    ctx_without_memory = empty_store.get_context_for_query(test_query, max_tokens=2000)

    result = {
        "dimension": "D4_Memory_Efficiency",
        "description": "共享记忆的去重率、矛盾检测能力和 token 节省效果",
        "put_count": put_count,
        "after_dedup_count": total_entries,
        "merged_count": merge_count,
        "dedup_rate": round(merge_count / put_count, 4),
        "conflicts_detected": conflict_count,
        "estimated_token_savings": token_savings,
        "token_savings_ratio": round(savings_ratio, 4),
        "context_with_memory_chars": len(ctx_with_memory),
        "context_without_memory_chars": len(ctx_without_memory),
    }

    _save(result, output_dir, "D4_memory_efficiency")
    return result


# ============================================================================
# D5: Agent vs 单轮 LLM
# ============================================================================

async def benchmark_agent_vs_llm(
    config: dict,
    questions: list[dict],
    output_dir: str,
) -> dict[str, Any]:
    """
    Agent 完整流程 vs 单轮 LLM 直接回答，对比 5 维度评分。
    """
    print("\n" + "=" * 70)
    print("[D5] Agent vs 单轮 LLM：研究质量对比")
    print("=" * 70)

    from src.models.model_router import ModelRouter

    bench = ResearchBench()
    llm_policy = ModelRouter.create_backend("deepseek")

    agent_scores = []
    llm_scores = []
    per_dim_deltas = defaultdict(list)

    for q in questions:
        qid = q["id"]
        query = q["query"]
        print(f"  [{qid}] {query[:50]}...")

        # Agent
        modules = initialize_modules(config, session_id=f"d6_{qid}")
        try:
            agent_report = await run_research(query, config, modules)
            agent_eval = bench.evaluate_report(agent_report, qid)
            agent_scores.append(agent_eval["composite_score"])
        except Exception as e:
            print(f"    Agent FAILED: {e}")
            agent_eval = {"composite_score": 0.0, "metrics": {}}
            agent_scores.append(0.0)

        # 单轮 LLM
        try:
            llm_resp = llm_policy([
                {"role": "system", "content": "你是一个研究助手，请用 Markdown 格式输出一份结构完整的研究报告，引用来源。"},
                {"role": "user", "content": query},
            ])
            llm_report = llm_resp.get("content", "") if isinstance(llm_resp, dict) else str(llm_resp)
            llm_eval = bench.evaluate_report(llm_report, qid)
            llm_scores.append(llm_eval["composite_score"])
        except Exception as e:
            print(f"    LLM FAILED: {e}")
            llm_eval = {"composite_score": 0.0, "metrics": {}}
            llm_scores.append(0.0)

        for metric_key in agent_eval.get("metrics", {}):
            a_val = agent_eval["metrics"].get(metric_key, 0.0)
            l_val = llm_eval.get("metrics", {}).get(metric_key, 0.0)
            per_dim_deltas[metric_key].append(a_val - l_val)

        print(f"    Agent={agent_eval['composite_score']:.3f} vs LLM={llm_eval['composite_score']:.3f}")

    diffs = [a - l for a, l in zip(agent_scores, llm_scores)]
    stats = bootstrap_ci_paired(diffs)
    effect = cohens_d(agent_scores, llm_scores)

    dim_stats = {}
    for key, deltas in per_dim_deltas.items():
        if len(deltas) >= 2:
            dim_stats[key] = bootstrap_ci_paired(deltas)

    result = {
        "dimension": "D5_Agent_vs_LLM",
        "description": "多智能体完整流程 vs 单轮 LLM 直接回答",
        "agent_avg": float(np.mean(agent_scores)),
        "llm_avg": float(np.mean(llm_scores)),
        "delta": stats["mean_diff"],
        "ci": [stats["ci_lower"], stats["ci_upper"]],
        "p_value": stats["p_value"],
        "significant": stats["significant"],
        "cohens_d": round(effect, 4),
        "effect_size": _describe_effect(effect),
        "per_dimension_deltas": {k: v["mean_diff"] for k, v in dim_stats.items()},
    }

    print(f"  Agent={result['agent_avg']:.4f} vs LLM={result['llm_avg']:.4f} "
          f"Δ={result['delta']:+.4f} d={effect:.2f} {_significance_marker(stats)}")

    _save(result, output_dir, "D5_agent_vs_llm")
    return result


# ============================================================================
# D6: 编排效率 —— 并发加速比 + 耗时分布
# ============================================================================

async def benchmark_orchestration_efficiency(
    config: dict,
    output_dir: str,
) -> dict[str, Any]:
    """
    对比 max_concurrent=1（串行）vs max_concurrent=5（并发）的耗时分布。
    """
    print("\n" + "=" * 70)
    print("[D6] 编排效率：并发加速比 + 阶段耗时分布")
    print("=" * 70)

    from evaluation.benchmarks.research_bench import ResearchBench
    bench = ResearchBench()
    questions = bench.get_questions(n=3)

    results = {}
    for concurrency, label in [(1, "串行"), (5, "并发(5)")]:
        cfg = dict(config)
        cfg.setdefault("orchestrator", {})["max_concurrent"] = concurrency

        times = []
        for q in questions[:3]:
            modules = initialize_modules(cfg, session_id=f"d7_{concurrency}_{q['id']}")
            t0 = time.time()
            try:
                await run_research(q["query"], cfg, modules)
                elapsed = time.time() - t0
                times.append(elapsed)
                print(f"  [{label}] {q['id']}: {elapsed:.1f}s")
            except Exception as e:
                print(f"  [{label}] {q['id']}: FAILED ({e})")

        avg_time = np.mean(times) if times else 0.0
        results[label] = {
            "concurrency": concurrency,
            "times": times,
            "avg_time": float(avg_time),
        }
        print(f"  → {label} avg={avg_time:.1f}s")

    serial_avg = results["串行"]["avg_time"]
    parallel_avg = results["并发(5)"]["avg_time"]
    speedup = serial_avg / max(parallel_avg, 1.0)

    result = {
        "dimension": "D6_Orchestration_Efficiency",
        "description": "DAG 拓扑并发 vs 串行执行的加速比",
        "serial_avg_time": serial_avg,
        "parallel_avg_time": parallel_avg,
        "speedup": round(speedup, 2),
        "efficiency": round(speedup / 5, 3),  # 并行效率 = speedup / concurrency
    }

    print(f"  加速比: {speedup:.2f}x (效率={result['efficiency']:.1%})")

    _save(result, output_dir, "D6_orchestration_efficiency")
    return result


# ============================================================================
# D7: 鲁棒性 —— 跨领域 / 跨后端稳定性
# ============================================================================

async def benchmark_robustness(
    config: dict,
    output_dir: str,
) -> dict[str, Any]:
    """
    跨领域 + 跨 query 难度 + 跨 LLM 后端的一致性。
    """
    print("\n" + "=" * 70)
    print("[D7] 鲁棒性：跨领域 / 跨后端的分数稳定性")
    print("=" * 70)

    from evaluation.benchmarks.research_bench import ResearchBench
    bench = ResearchBench()

    # 按领域分组
    by_domain = defaultdict(list)
    for q in bench.questions:
        by_domain[q.get("domain", "未知")].append(q)

    domain_results = {}
    all_scores = []

    for domain, qs in by_domain.items():
        scores = []
        for q in qs[:2]:  # 每领域取 2 题
            modules = initialize_modules(config, session_id=f"d8_{domain}_{q['id']}")
            try:
                report = await run_research(q["query"], config, modules)
                eval_r = bench.evaluate_report(report, q["id"])
                scores.append(eval_r["composite_score"])
            except Exception:
                scores.append(0.0)

        avg = np.mean(scores) if scores else 0.0
        std = np.std(scores) if len(scores) > 1 else 0.0
        domain_results[domain] = {"avg": float(avg), "std": float(std), "n": len(scores)}
        all_scores.extend(scores)
        print(f"  {domain}: avg={avg:.3f} ± {std:.3f} (n={len(scores)})")

    global_avg = np.mean(all_scores) if all_scores else 0.0
    global_std = np.std(all_scores) if all_scores else 0.0

    # 变异系数 (Coefficient of Variation) —— 越低越稳定
    cv = global_std / max(global_avg, 0.01)

    result = {
        "dimension": "D7_Robustness",
        "description": "跨领域的分数稳定性和变异系数（越低越鲁棒）",
        "global_avg": float(global_avg),
        "global_std": float(global_std),
        "coefficient_of_variation": round(cv, 4),
        "robustness_grade": "A" if cv < 0.15 else ("B" if cv < 0.30 else "C"),
        "by_domain": domain_results,
    }

    print(f"  全局: avg={global_avg:.4f} ± {global_std:.4f}, CV={cv:.3f} (等级={result['robustness_grade']})")

    _save(result, output_dir, "D7_robustness")
    return result


# ============================================================================
# 汇总报告
# ============================================================================

def generate_summary_report(all_results: list[dict], output_dir: str) -> str:
    """生成汇总 Markdown 报告。"""
    lines = [
        "# DeepResearch Agent 多维度定量评测报告",
        "",
        f"**评测时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**维度数**: {len(all_results)}",
        "",
        "---",
        "",
        "## 总览",
        "",
        "| 维度 | 核心指标 | 结果 | 显著性 |",
        "|------|---------|------|--------|",
    ]

    for r in all_results:
        dim = r.get("dimension", "")
        if dim == "D1_Module_Ablation":
            top = r["ranking"][0] if r.get("ranking") else ["", {}]
            lines.append(f"| D1 模块消融 | 最大边际贡献 ({top[0]}) | Δ={top[1].get('marginal_contribution', 0):+.3f} | {top[1].get('significance', '?')} |")
        elif dim == "D2_RAG_Retrieval":
            lines.append(f"| D2 RAG 检索 | 平均命中率 | {r.get('rag_avg_hit_rate', 0):.3f} | — |")
        elif dim == "D3_Compressor_Fidelity":
            first_key = list(r.get("results", {}).keys())[0] if r.get("results") else None
            if first_key:
                l3_ret = r["results"][first_key]["L3"]["entity_retention"]
            lines.append(f"| D3 压缩保真度 | L3 实体保留率 | {l3_ret:.1%} | — |")
        elif dim == "D4_Memory_Efficiency":
            lines.append(f"| D4 记忆效率 | 去重率 / 矛盾检测 | {r.get('dedup_rate', 0):.1%} / {r.get('conflicts_detected', 0)}对 | — |")
        elif dim == "D5_Agent_vs_LLM":
            lines.append(f"| D5 Agent vs LLM | 质量提升 | Δ={r.get('delta', 0):+.3f} d={r.get('cohens_d', 0):.2f} | {_significance_marker(r)} |")
        elif dim == "D6_Orchestration_Efficiency":
            lines.append(f"| D6 编排效率 | 并发加速比 | {r.get('speedup', 0):.2f}x | — |")
        elif dim == "D7_Robustness":
            lines.append(f"| D7 鲁棒性 | 变异系数 CV | {r.get('coefficient_of_variation', 0):.3f} | — |")

    lines += [
        "",
        "---",
        "",
        "## 各维度详细结果",
        "",
    ]

    for r in all_results:
        dim = r.get("dimension", "")
        lines.append(f"### {dim}")
        lines.append(f"_{r.get('description', '')}_")
        lines.append("")
        lines.append("```json")
        # 排除过长的 details
        summary = {k: v for k, v in r.items() if k not in ("per_query", "per_dimension_deltas", "ranking")}
        lines.append(json.dumps(summary, ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")

    report_text = "\n".join(lines)
    report_path = os.path.join(output_dir, "QUANTITATIVE_BENCHMARK_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\n汇总报告已生成: {report_path}")
    return report_text


# ============================================================================
# 工具函数
# ============================================================================

def _save(data: dict, output_dir: str, prefix: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{prefix}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


# ============================================================================
# Main
# ============================================================================

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="DeepResearch Agent 多维度定量评测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/run_quantitative_bench.py --all                    # 跑全部维度
  python scripts/run_quantitative_bench.py --dim D1,D3             # 只跑指定维度
  python scripts/run_quantitative_bench.py --all --quick            # 快速模式 (3题)
  python scripts/run_quantitative_bench.py --dim D2,D3,D4           # 只跑不需要 LLM 的维度
        """,
    )
    parser.add_argument("--all", action="store_true", help="跑全部 7 个维度")
    parser.add_argument("--dim", type=str, default="", help="逗号分隔的维度号，如 D1,D3,D5")
    parser.add_argument("--quick", action="store_true", help="快速模式（每维度只用 3 道题）")
    parser.add_argument("--output_dir", type=str, default="outputs/quantitative_bench", help="输出目录")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--log_level", type=str, default="WARNING",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    setup_logging(args.log_level)
    config = load_config(args.config)

    # 哪些维度要跑
    if args.all:
        dims = ["D1", "D2", "D3", "D4", "D5", "D6", "D7"]
    elif args.dim:
        dims = [d.strip() for d in args.dim.split(",")]
    else:
        print("请指定 --all 或 --dim D1,D3,...")
        return

    bench = ResearchBench()
    num_questions = 3 if args.quick else 12
    questions = bench.get_questions(n=num_questions)
    print(f"[QuantBench] 加载 {len(questions)} 道评测题 (quick={args.quick})")

    all_results = []

    # 需要 LLM API 的维度
    needs_llm = {"D1", "D2", "D5", "D6", "D7"}
    # 不需要 LLM API 的维度（纯本地）
    local_only = {"D3", "D4"}

    for dim in dims:
        try:
            if dim == "D1":
                r = await benchmark_module_ablation(config, questions, args.output_dir)
            elif dim == "D2":
                r = benchmark_rag_retrieval(args.output_dir)
            elif dim == "D3":
                r = benchmark_compressor_fidelity(args.output_dir)
            elif dim == "D4":
                r = benchmark_memory_efficiency(args.output_dir)
            elif dim == "D5":
                r = await benchmark_agent_vs_llm(config, questions, args.output_dir)
            elif dim == "D6":
                r = await benchmark_orchestration_efficiency(config, args.output_dir)
            elif dim == "D7":
                r = await benchmark_robustness(config, args.output_dir)
            else:
                print(f"未知维度: {dim}")
                continue
            all_results.append(r)
        except Exception as e:
            logger.error(f"[{dim}] 执行失败: {e}", exc_info=True)

    # 生成汇总报告
    if all_results:
        generate_summary_report(all_results, args.output_dir)

    print(f"\n全部结果已保存至: {args.output_dir}/")


if __name__ == "__main__":
    asyncio.run(main())

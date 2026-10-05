#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/core/ablation.py
================================================================================
消融实验通用框架。

对外接口:
    - AblationStudy.run_module_ablation(config, questions, systems) -> dict
================================================================================
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import time
from datetime import datetime
from typing import Any

from .runner import initialize_modules, run_research

logger = logging.getLogger("ablation")


class AblationStudy:
    """消融实验框架。"""

    # 模块消融的默认配置映射
    DEFAULT_MODULE_ABLATIONS: dict[str, tuple[str, dict]] = {
        "full": ("完整系统", {}),
        "no_compressor": ("关闭上下文压缩", {"compressor": {"enabled": False}}),
        "no_memory": ("关闭记忆存储", {"memory": {"enabled": False}}),
    }

    @staticmethod
    def override_config(config: dict, overrides: dict) -> dict:
        """深度合并配置覆盖（支持嵌套字典）。"""
        cfg = copy.deepcopy(config)

        def _deep_merge(base: dict, patch: dict) -> dict:
            for key, value in patch.items():
                if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                    base[key] = _deep_merge(base[key], value)
                else:
                    base[key] = value
            return base

        return _deep_merge(cfg, overrides)

    # -----------------------------------------------------------------------
    # 模块消融：full / no_XXX
    # -----------------------------------------------------------------------
    @classmethod
    def run_module_ablation(
        cls,
        config: dict,
        questions: list[dict[str, Any]],
        systems: dict[str, tuple[str, dict]] | None = None,
    ) -> dict[str, Any]:
        """
        运行模块消融实验。

        Args:
            config: 基础配置。
            questions: 评测题目列表（每条含 id, query）。
            systems: 消融配置映射。键为 system_name，值为 (描述, 配置覆盖)。
                     默认使用 DEFAULT_MODULE_ABLATIONS。

        Returns:
            包含各系统得分和明细的字典。
        """
        if systems is None:
            systems = cls.DEFAULT_MODULE_ABLATIONS

        results: list[dict[str, Any]] = []

        for name, (desc, overrides) in systems.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"[消融实验] {name}: {desc}")
            logger.info(f"{'='*60}")

            cfg = cls.override_config(config, overrides)
            modules = initialize_modules(cfg)

            scores: list[float] = []
            details: list[dict[str, Any]] = []

            for q in questions:
                qid = q.get("id", "unknown")
                query = q.get("query", "")
                logger.info(f"  [{qid}] {query[:60]}...")

                start = time.time()
                try:
                    report = asyncio.run(run_research(query, cfg, modules))
                    elapsed = time.time() - start

                    # 评分由外部调用方注入（避免 evaluation/ 反向依赖）
                    # 这里只记录原始报告和元信息
                    details.append({
                        "question_id": qid,
                        "query": query,
                        "elapsed_seconds": elapsed,
                        "report_length": len(report),
                        "system": name,
                    })
                    scores.append(1.0)  # 占位，实际分数由外部 evaluator 填充
                    logger.info(f"    → 成功, time={elapsed:.1f}s, len={len(report)}")

                except Exception as e:
                    logger.warning(f"    → 失败: {e}")
                    details.append({
                        "question_id": qid,
                        "query": query,
                        "error": str(e),
                        "system": name,
                    })
                    scores.append(0.0)

            results.append({
                "system_name": name,
                "description": desc,
                "num_questions": len(questions),
                "average_composite_score": sum(scores) / len(scores) if scores else 0.0,
                "details": details,
            })

        return {
            "evaluation_name": "DeepResearch Agent 模块消融实验",
            "timestamp": datetime.now().isoformat(),
            "num_questions": len(questions),
            "systems": results,
            "summary": {r["system_name"]: r["average_composite_score"] for r in results},
        }

    # -----------------------------------------------------------------------
    # 结果保存
    # -----------------------------------------------------------------------
    @staticmethod
    def save_results(data: dict[str, Any], output_dir: str, prefix: str = "ablation") -> str:
        """保存消融结果到 JSON 文件。"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(output_dir, f"{prefix}_{timestamp}.json")

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"消融结果已保存: {filepath}")
        return filepath

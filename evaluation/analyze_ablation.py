#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation/analyze_ablation.py
================================================================================
消融实验分析脚本。

分析以下维度的消融结果：
1. 对抗降噪轮数：0轮 vs 1轮 vs 2轮 vs 3轮
2. 其他模块（可选）

支持绘制对比柱状图、折线图，并输出统计报告。
================================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # 无 GUI 环境使用 Agg 后端

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class AblationAnalyzer:
    """消融实验结果分析器。"""

    def __init__(self, output_dir: str = "outputs/evaluation") -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # 数据加载
    # -----------------------------------------------------------------------
    @staticmethod
    def load_results(path: str) -> dict[str, Any]:
        """从 JSON 文件加载评测结果。"""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # -----------------------------------------------------------------------
    # 对抗轮数消融分析
    # -----------------------------------------------------------------------
    def analyze_adversarial_rounds(
        self,
        results: dict[str, Any],
        save_prefix: str = "ablation_adversarial",
    ) -> dict[str, Any]:
        """
        分析对抗降噪轮数对综合得分的影响。

        Args:
            results: 评测结果字典，键应包含 "adv_0", "adv_1", "adv_2", "adv_3" 等。
            save_prefix: 保存图片的文件名前缀。

        Returns:
            统计分析结果字典。
        """
        rounds = []
        scores = []

        for key in sorted(results.keys()):
            if key.startswith("adv_"):
                r = int(key.split("_")[1])
                rounds.append(r)
                scores.append(results[key])

        if not rounds:
            print("[AblationAnalyzer] 未找到对抗轮数数据")
            return {}

        # 绘制折线图
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(rounds, scores, marker="o", linewidth=2, markersize=8, color="#2E86AB")
        ax.set_xlabel("Adversarial Rounds", fontsize=12)
        ax.set_ylabel("Average Composite Score", fontsize=12)
        ax.set_title("Effect of Adversarial Rounds on Report Quality", fontsize=14)
        ax.set_xticks(rounds)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_ylim([0.0, 1.0])

        save_path = os.path.join(self.output_dir, f"{save_prefix}.png")
        fig.tight_layout()
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"[AblationAnalyzer] 图表已保存: {save_path}")

        return {
            "dimension": "adversarial_rounds",
            "rounds": rounds,
            "scores": scores,
            "best_round": rounds[np.argmax(scores)],
            "best_score": max(scores),
        }

    # -----------------------------------------------------------------------
    # 综合消融报告
    # -----------------------------------------------------------------------
    def generate_report(
        self,
        adversarial_results: dict[str, Any] | None = None,
        output_name: str = "ablation_analysis.json",
    ) -> str:
        """
        生成综合消融分析报告。

        Args:
            adversarial_results: 对抗消融结果。
            output_name: 输出 JSON 文件名。

        Returns:
            保存的文件路径。
        """
        report = {
            "adversarial": adversarial_results or {},
        }

        path = os.path.join(self.output_dir, output_name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"[AblationAnalyzer] 消融分析报告已保存: {path}")
        return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="消融实验分析脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python evaluation/analyze_ablation.py --adv_results outputs/evaluation/adv_results.json
        """,
    )
    parser.add_argument(
        "--adv_results",
        type=str,
        default=None,
        help="对抗轮数消融结果 JSON 文件路径（格式：{\"adv_0\": 0.6, \"adv_1\": 0.72, ...}）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/evaluation",
        help="图表和报告输出目录",
    )
    args = parser.parse_args()

    analyzer = AblationAnalyzer(output_dir=args.output_dir)

    adv_report = None

    if args.adv_results and os.path.exists(args.adv_results):
        adv_data = analyzer.load_results(args.adv_results)
        adv_report = analyzer.analyze_adversarial_rounds(adv_data)

    # 如果没有提供文件，生成示例数据进行演示
    if adv_report is None:
        print("[main] 未提供输入数据，使用示例数据生成演示图表...")
        adv_report = analyzer.analyze_adversarial_rounds(
            {"adv_0": 0.62, "adv_1": 0.71, "adv_2": 0.78, "adv_3": 0.80}
        )

    analyzer.generate_report(adv_report)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agent vs single-turn LLM on a sampled DeepResearch Bench subset."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import random
import re
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.benchmarks.deep_research_bench import DeepResearchBench
from evaluation.metrics.rule_based import RuleBasedMetrics
from evaluation.metrics.stats import bootstrap_ci_paired, cohens_d
from src.core.runner import initialize_modules, load_config, run_research, setup_logging
from src.models.model_router import ModelRouter


BASELINE_SYSTEM_PROMPT = """你是一名严谨的研究助手。请直接回答用户问题，不得调用工具或假装已经浏览网页。
输出一份结构完整的 Markdown 研究报告，包含问题界定、核心分析、局限性和结论。
只引用你能够明确给出来源名称或 URL 的材料；不确定的事实必须明确标注不确定性。
不要输出自评置信度。"""

PROXY_DIMENSIONS = (
    "citation_coverage",
    "logical_consistency",
    "low_hallucination_proxy",
)
JUDGE_DIMENSIONS = (
    "comprehensiveness",
    "insight",
    "instruction_following",
    "readability",
)


def _module_sampling(config: dict[str, Any], module_name: str, backend_name: str) -> dict[str, Any]:
    sampling = config.get("model", {}).get("backend_sampling", {})
    result = dict(sampling.get(backend_name, {}))
    result.update(sampling.get("modules", {}).get(module_name, {}))
    return result


def _baseline_policy_config(
    config: dict[str, Any],
    backend_override: str | None,
    temperature_override: float | None,
    max_tokens_override: int | None,
) -> tuple[str, dict[str, Any]]:
    model_cfg = config.get("model", {})
    mapping = model_cfg.get("backend_mapping", {})
    backend = backend_override or mapping.get("summarizer") or mapping.get("solver") or model_cfg.get("backend", "deepseek")

    # A one-shot baseline gets the same final-answer budget as the Agent writer.
    sampling = _module_sampling(config, "summarizer", backend)
    if temperature_override is not None:
        sampling["temperature"] = temperature_override
    if max_tokens_override is not None:
        sampling["max_tokens"] = max_tokens_override
    return backend, sampling


def run_baseline(
    query: str,
    config: dict[str, Any],
    backend_override: str | None = None,
    temperature_override: float | None = None,
    max_tokens_override: int | None = None,
) -> dict[str, Any]:
    backend, sampling = _baseline_policy_config(
        config, backend_override, temperature_override, max_tokens_override
    )
    policy = ModelRouter.create_backend(backend, **sampling)
    started = time.perf_counter()
    response = policy([
        {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ])
    elapsed = time.perf_counter() - started
    content = str(response.get("content", "")) if isinstance(response, dict) else str(response)
    status = "failed" if not content.strip() or content.lstrip().startswith("Error:") else "success"
    return {
        "system": "baseline",
        "status": status,
        "content": content,
        "elapsed_seconds": round(elapsed, 3),
        "token_usage": response.get("usage", {}) if isinstance(response, dict) else {},
        "token_usage_kind": "api_reported",
        "model_backend": backend,
        "sampling": sampling,
    }


_META_PATTERNS = {
    "confidence": re.compile(r"\*\*置信度\*\*:\s*([0-9.]+)"),
    "num_searches": re.compile(r"\*\*搜索轮数\*\*:\s*(\d+)"),
    "num_replan": re.compile(r"\*\*重规划次数\*\*:\s*(\d+)"),
    "adversarial_rounds": re.compile(r"\*\*对抗轮数\*\*:\s*(\d+)"),
    "estimated_tokens": re.compile(r"\*\*估算 Token\*\*:\s*(\d+)"),
}


def _parse_agent_metadata(report: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key, pattern in _META_PATTERNS.items():
        match = pattern.search(report)
        if match:
            metadata[key] = float(match.group(1)) if key == "confidence" else int(match.group(1))
    return metadata


async def run_agent(query: str, config: dict[str, Any], session_id: str) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        modules = initialize_modules(copy.deepcopy(config), session_id=session_id)
        content = await run_research(query, config, modules)
        status = "success"
        if not content.strip() or "Report generation failed unexpectedly" in content:
            status = "failed"
        error = None
    except Exception as exc:
        content = ""
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started
    metadata = _parse_agent_metadata(content)
    estimated_tokens = int(metadata.get("estimated_tokens", 0))
    return {
        "system": "agent",
        "status": status,
        "content": content,
        "error": error,
        "elapsed_seconds": round(elapsed, 3),
        "token_usage": {"total_tokens": estimated_tokens},
        "token_usage_kind": "application_estimate",
        "metadata": metadata,
    }


def _failed_proxy_evaluation(question_id: str, topic: str) -> dict[str, Any]:
    metrics = {key: 0.0 for key in PROXY_DIMENSIONS}
    return {
        "question_id": question_id,
        "topic": topic,
        "metrics": metrics,
        "composite_score": 0.0,
        "hallucination_rate": 1.0,
        "score_kind": "diagnostic_proxy",
    }


def evaluate_proxies(
    question: dict[str, Any],
    run: dict[str, Any],
) -> dict[str, Any]:
    if run["status"] != "success":
        return _failed_proxy_evaluation(question["id"], question["topic"])
    report = run["content"]
    hallucination_rate = RuleBasedMetrics.hallucination_rate(report)
    metrics = {
        "citation_coverage": RuleBasedMetrics.citation_coverage(report),
        "logical_consistency": RuleBasedMetrics.logical_consistency(report),
        "low_hallucination_proxy": max(0.0, 1.0 - hallucination_rate),
    }
    # These deterministic signals are diagnostics only. DeepResearch Bench
    # quality is judged with its official task-specific RACE criteria.
    composite = (
        0.50 * metrics["citation_coverage"]
        + 0.25 * metrics["logical_consistency"]
        + 0.25 * metrics["low_hallucination_proxy"]
    )
    return {
        "question_id": question["id"],
        "topic": question["topic"],
        "metrics": metrics,
        "composite_score": composite,
        "hallucination_rate": hallucination_rate,
        "score_kind": "diagnostic_proxy",
    }


def _judge_one_pass(
    judge: Any,
    question: dict[str, Any],
    first_system: str,
    reports: dict[str, str],
) -> dict[str, Any]:
    second_system = "agent" if first_system == "baseline" else "baseline"
    raw = judge.compare_two_race(
        reports[first_system],
        reports[second_system],
        question["query"],
        criteria=question["criteria"],
        reference_report=question["reference_article"],
    )
    mapped: dict[str, Any] = {}
    if "error" not in raw:
        for dimension in JUDGE_DIMENSIONS:
            values = raw.get(dimension, {})
            if isinstance(values, dict) and "A" in values and "B" in values:
                mapped[dimension] = {
                    first_system: float(values["A"]),
                    second_system: float(values["B"]),
                    "reason": values.get("reason", ""),
                }
    return {
        "label_map": {"A": first_system, "B": second_system},
        "raw": raw,
        "mapped": mapped,
    }


def evaluate_with_judge(
    judge: Any,
    question: dict[str, Any],
    reports: dict[str, str],
    first_system: str,
    passes: int,
) -> dict[str, Any]:
    pass_results = []
    order = first_system
    for _ in range(passes):
        pass_results.append(_judge_one_pass(judge, question, order, reports))
        order = "agent" if order == "baseline" else "baseline"

    aggregate: dict[str, Any] = {}
    for dimension in JUDGE_DIMENSIONS:
        baseline_scores = []
        agent_scores = []
        for result in pass_results:
            values = result["mapped"].get(dimension, {})
            if "baseline" in values and "agent" in values:
                baseline_scores.append(values["baseline"])
                agent_scores.append(values["agent"])
        if baseline_scores:
            baseline_avg = statistics.fmean(baseline_scores)
            agent_avg = statistics.fmean(agent_scores)
            aggregate[dimension] = {
                "baseline": round(baseline_avg, 4),
                "agent": round(agent_avg, 4),
                "delta": round(agent_avg - baseline_avg, 4),
            }
    weights = question["criteria"].get("dimension_weight", {})
    available = [dimension for dimension in JUDGE_DIMENSIONS if dimension in aggregate]
    weight_total = sum(float(weights.get(dimension, 0.0)) for dimension in available)
    if available:
        if weight_total <= 0:
            normalized = {dimension: 1.0 / len(available) for dimension in available}
        else:
            normalized = {
                dimension: float(weights.get(dimension, 0.0)) / weight_total
                for dimension in available
            }
        baseline_overall = sum(
            aggregate[dimension]["baseline"] * normalized[dimension]
            for dimension in available
        )
        agent_overall = sum(
            aggregate[dimension]["agent"] * normalized[dimension]
            for dimension in available
        )
        aggregate["overall"] = {
            "baseline": round(baseline_overall, 4),
            "agent": round(agent_overall, 4),
            "delta": round(agent_overall - baseline_overall, 4),
        }
    return {"passes": pass_results, "aggregate": aggregate}


def _safe_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _append_jsonl(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")


def _save_report(run_dir: Path, question_id: str, repeat: int, run: dict[str, Any]) -> str:
    path = run_dir / "reports" / question_id / f"repeat_{repeat:02d}_{run['system']}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(run["content"], encoding="utf-8")
    return str(path.relative_to(run_dir))


def _question_level_pairs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["question_id"]].append(record)

    pairs = []
    for question_id, items in grouped.items():
        baseline = statistics.fmean(
            item["baseline"]["proxy_eval"]["composite_score"] for item in items
        )
        agent = statistics.fmean(
            item["agent"]["proxy_eval"]["composite_score"] for item in items
        )
        judge_values = [
            item["judge"]["aggregate"]["overall"]
            for item in items
            if (item.get("judge") or {}).get("aggregate", {}).get("overall")
        ]
        judge_overall = None
        if judge_values:
            judge_baseline = statistics.fmean(value["baseline"] for value in judge_values)
            judge_agent = statistics.fmean(value["agent"] for value in judge_values)
            judge_overall = {
                "baseline": judge_baseline,
                "agent": judge_agent,
                "delta": judge_agent - judge_baseline,
            }
        pairs.append({
            "question_id": question_id,
            "source_id": items[0]["source_id"],
            "topic": items[0]["topic"],
            "language": items[0]["language"],
            "proxy": {
                "baseline": baseline,
                "agent": agent,
                "delta": agent - baseline,
            },
            "judge_overall": judge_overall,
        })
    return pairs


def build_summary(records: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    pairs = _question_level_pairs(records)
    diffs = [pair["proxy"]["delta"] for pair in pairs]
    baseline_scores = [pair["proxy"]["baseline"] for pair in pairs]
    agent_scores = [pair["proxy"]["agent"] for pair in pairs]
    stats = bootstrap_ci_paired(diffs, seed=seed)
    effect = cohens_d(agent_scores, baseline_scores) if len(pairs) >= 2 else 0.0

    dimension_summary: dict[str, Any] = {}
    for dimension in PROXY_DIMENSIONS:
        per_question: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: {"baseline": [], "agent": []}
        )
        for record in records:
            qid = record["question_id"]
            for system in ("baseline", "agent"):
                value = record[system]["proxy_eval"]["metrics"].get(dimension, 0.0)
                per_question[qid][system].append(float(value))
        dim_diffs = [
            statistics.fmean(values["agent"]) - statistics.fmean(values["baseline"])
            for values in per_question.values()
        ]
        dimension_summary[dimension] = bootstrap_ci_paired(dim_diffs, seed=seed)

    judge_dimensions: dict[str, Any] = {}
    for dimension in (*JUDGE_DIMENSIONS, "overall"):
        deltas_by_question: dict[str, list[float]] = defaultdict(list)
        for record in records:
            aggregate = (record.get("judge") or {}).get("aggregate", {})
            if dimension in aggregate:
                deltas_by_question[record["question_id"]].append(aggregate[dimension]["delta"])
        if deltas_by_question:
            judge_dimensions[dimension] = bootstrap_ci_paired(
                [statistics.fmean(values) for values in deltas_by_question.values()], seed=seed
            )

    efficiency: dict[str, Any] = {}
    for system in ("baseline", "agent"):
        runs = [record[system] for record in records]
        elapsed = [float(run["elapsed_seconds"]) for run in runs]
        tokens = [int(run.get("token_usage", {}).get("total_tokens", 0) or 0) for run in runs]
        efficiency[system] = {
            "success_rate": sum(run["status"] == "success" for run in runs) / max(len(runs), 1),
            "median_elapsed_seconds": statistics.median(elapsed) if elapsed else 0.0,
            "mean_total_tokens": statistics.fmean(tokens) if tokens else 0.0,
            "token_usage_kind": runs[0].get("token_usage_kind", "unknown") if runs else "unknown",
        }

    judge_pairs = [pair["judge_overall"] for pair in pairs if pair["judge_overall"]]
    judge_overall = None
    if judge_pairs:
        judge_diffs = [pair["delta"] for pair in judge_pairs]
        judge_baseline = [pair["baseline"] for pair in judge_pairs]
        judge_agent = [pair["agent"] for pair in judge_pairs]
        judge_stats = bootstrap_ci_paired(judge_diffs, seed=seed)
        judge_overall = {
            "num_questions": len(judge_pairs),
            "baseline_avg": statistics.fmean(judge_baseline),
            "agent_avg": statistics.fmean(judge_agent),
            "mean_delta": judge_stats["mean_diff"],
            "ci_95": [judge_stats["ci_lower"], judge_stats["ci_upper"]],
            "p_value_one_sided": judge_stats["p_value"],
            "significant": judge_stats["significant"],
            "cohens_d": round(
                cohens_d(judge_agent, judge_baseline) if len(judge_pairs) >= 2 else 0.0,
                4,
            ),
            "wins": sum(diff > 0 for diff in judge_diffs),
            "ties": sum(abs(diff) < 1e-12 for diff in judge_diffs),
            "losses": sum(diff < 0 for diff in judge_diffs),
        }

    return {
        "benchmark": "DeepResearch Bench",
        "num_questions": len(pairs),
        "num_paired_runs": len(records),
        "judge_overall": judge_overall,
        "diagnostic_proxy_composite": {
            "baseline_avg": statistics.fmean(baseline_scores) if baseline_scores else 0.0,
            "agent_avg": statistics.fmean(agent_scores) if agent_scores else 0.0,
            "mean_delta": stats["mean_diff"],
            "ci_95": [stats["ci_lower"], stats["ci_upper"]],
            "p_value_one_sided": stats["p_value"],
            "significant": stats["significant"],
            "cohens_d": round(effect, 4),
            "wins": sum(diff > 0 for diff in diffs),
            "ties": sum(abs(diff) < 1e-12 for diff in diffs),
            "losses": sum(diff < 0 for diff in diffs),
        },
        "diagnostic_proxy_dimensions": dimension_summary,
        "judge_dimensions": judge_dimensions,
        "efficiency": efficiency,
        "per_question": pairs,
    }


def render_summary_markdown(summary: dict[str, Any]) -> str:
    proxy = summary["diagnostic_proxy_composite"]
    judge = summary.get("judge_overall")
    efficiency = summary["efficiency"]
    lines = [
        "# DeepResearch Bench: Agent vs Single-turn LLM",
        "",
        f"- Questions: {summary['num_questions']}",
        f"- Paired runs: {summary['num_paired_runs']}",
        "",
        "## RACE-aligned anonymous Judge",
        "",
    ]
    if judge:
        lines.extend([
            "| System | Weighted score (1-5) |",
            "|---|---:|",
            f"| Single-turn LLM | {judge['baseline_avg']:.4f} |",
            f"| DeepResearch Agent | {judge['agent_avg']:.4f} |",
            "",
            (
                f"Mean paired delta: **{judge['mean_delta']:+.4f}**, "
                f"95% CI [{judge['ci_95'][0]:+.4f}, {judge['ci_95'][1]:+.4f}], "
                f"W/T/L = {judge['wins']}/{judge['ties']}/{judge['losses']}."
            ),
        ])
    else:
        lines.append("Judge was skipped or produced no valid paired scores.")
    lines.extend([
        "",
        "## Diagnostic proxies and efficiency",
        "",
        "| System | Proxy composite | Success rate | Median latency (s) | Mean tokens |",
        "|---|---:|---:|---:|---:|",
        (
            f"| Single-turn LLM | {proxy['baseline_avg']:.4f} | "
            f"{efficiency['baseline']['success_rate']:.1%} | "
            f"{efficiency['baseline']['median_elapsed_seconds']:.1f} | "
            f"{efficiency['baseline']['mean_total_tokens']:.0f} |"
        ),
        (
            f"| DeepResearch Agent | {proxy['agent_avg']:.4f} | "
            f"{efficiency['agent']['success_rate']:.1%} | "
            f"{efficiency['agent']['median_elapsed_seconds']:.1f} | "
            f"{efficiency['agent']['mean_total_tokens']:.0f}* |"
        ),
        "",
        (
            f"Proxy delta: **{proxy['mean_delta']:+.4f}**, "
            f"95% CI [{proxy['ci_95'][0]:+.4f}, {proxy['ci_95'][1]:+.4f}]."
        ),
        "",
        "Proxy scores only measure citation presence, surface consistency and hallucination indicators; they are not official DeepResearch Bench scores.",
        "",
        "\\* Agent tokens are application estimates; baseline tokens are API-reported.",
    ])
    return "\n".join(lines) + "\n"


def select_questions(
    bench: DeepResearchBench, args: argparse.Namespace
) -> list[dict[str, Any]]:
    if args.ids:
        requested = [
            bench._normalize_source_id(value)
            for value in args.ids.split(",")
            if value.strip()
        ]
        questions = bench.get_by_source_ids(requested)
    elif args.suite == "demo":
        pool = bench.get_questions(language=args.language, topic=args.topic)
        if args.sample_size > len(pool):
            raise ValueError(
                f"Cannot sample {args.sample_size} tasks from a pool of {len(pool)}"
            )
        questions = random.Random(args.seed).sample(pool, args.sample_size)
    else:
        questions = bench.get_questions(language=args.language, topic=args.topic)
    if not questions:
        raise ValueError("No DeepResearch Bench tasks selected")
    return questions


async def run_benchmark(args: argparse.Namespace) -> Path:
    config = load_config(args.config)
    bench = DeepResearchBench(
        dataset_root=args.dataset_root,
        query_file=args.query_file,
        criteria_file=args.criteria_file,
        reference_file=args.reference_file,
    )
    questions = select_questions(bench, args)

    run_name = args.run_name or datetime.now().strftime("agent_vs_llm_%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Benchmark output directory is not empty: {run_dir}. "
            "Choose a new --run-name to avoid mixing runs."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid4().hex

    manifest = {
        "run_id": run_id,
        "run_name": run_name,
        "created_at": datetime.now().isoformat(),
        "benchmark": bench.dataset_metadata(),
        "suite": args.suite,
        "language": args.language,
        "topic": args.topic,
        "sample_size": len(questions),
        "seed": args.seed,
        "repeats": args.repeats,
        "judge_backend": None if args.skip_judge else args.judge_backend,
        "judge_passes": 0 if args.skip_judge else args.judge_passes,
        "question_ids": [q["id"] for q in questions],
        "source_question_ids": [q["source_id"] for q in questions],
        "config_path": args.config or "configs/default.yaml",
    }
    _safe_write_json(run_dir / "manifest.json", manifest)
    question_snapshot = []
    for question in questions:
        snapshot = copy.deepcopy(question)
        snapshot["reference_article_chars"] = len(snapshot["reference_article"])
        snapshot.pop("reference_article")
        question_snapshot.append(snapshot)
    _safe_write_json(run_dir / "questions.json", question_snapshot)
    _safe_write_json(run_dir / "config_snapshot.json", config)

    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return run_dir

    judge = None
    if not args.skip_judge:
        from src.core.judge import LLMJudge
        judge = LLMJudge(backend=args.judge_backend)

    records: list[dict[str, Any]] = []
    records_path = run_dir / "records.jsonl"
    total = len(questions) * args.repeats
    pair_index = 0

    for question in questions:
        for repeat in range(1, args.repeats + 1):
            pair_index += 1
            pair_seed = f"{args.seed}:{question['id']}:{repeat}"
            rng = random.Random(pair_seed)
            execution_order = ["baseline", "agent"]
            rng.shuffle(execution_order)
            print(f"\n[{pair_index}/{total}] {question['id']} repeat={repeat} order={execution_order}")

            runs: dict[str, dict[str, Any]] = {}
            for system in execution_order:
                if system == "baseline":
                    run = run_baseline(
                        question["query"],
                        config,
                        backend_override=args.baseline_backend,
                        temperature_override=args.baseline_temperature,
                        max_tokens_override=args.baseline_max_tokens,
                    )
                else:
                    # A unique run id prevents persistent memory from leaking
                    # evidence across independent benchmark executions.
                    session_id = f"avsl_{run_id}_{question['id']}_r{repeat}"
                    run = await run_agent(question["query"], config, session_id)
                run["report_path"] = _save_report(run_dir, question["id"], repeat, run)
                run["proxy_eval"] = evaluate_proxies(question, run)
                _append_jsonl(
                    run_dir / "official_format" / f"{system}_repeat_{repeat:02d}.jsonl",
                    {
                        "id": question["source_id"],
                        "prompt": question["query"],
                        "article": run["content"],
                    },
                )
                # The raw report is stored separately and omitted from JSONL.
                run.pop("content", None)
                runs[system] = run
                print(
                    f"  {system}: status={run['status']} "
                    f"proxy={run['proxy_eval']['composite_score']:.3f} "
                    f"time={run['elapsed_seconds']:.1f}s"
                )

            judge_result = None
            if judge is not None and runs["baseline"]["status"] == "success" and runs["agent"]["status"] == "success":
                reports = {
                    system: (run_dir / runs[system]["report_path"]).read_text(encoding="utf-8")
                    for system in ("baseline", "agent")
                }
                first_system = "baseline" if rng.random() < 0.5 else "agent"
                judge_result = evaluate_with_judge(
                    judge, question, reports, first_system, args.judge_passes
                )

            record = {
                "question_id": question["id"],
                "source_id": question["source_id"],
                "topic": question["topic"],
                "language": question["language"],
                "repeat": repeat,
                "execution_order": execution_order,
                "baseline": runs["baseline"],
                "agent": runs["agent"],
                "judge": judge_result,
            }
            records.append(record)
            _append_jsonl(records_path, record)

    summary = build_summary(records, seed=args.seed)
    _safe_write_json(run_dir / "summary.json", summary)
    (run_dir / "SUMMARY.md").write_text(render_summary_markdown(summary), encoding="utf-8")
    print(f"\nBenchmark complete: {run_dir}")
    headline = summary.get("judge_overall") or summary["diagnostic_proxy_composite"]
    print(json.dumps(headline, ensure_ascii=False, indent=2))
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DeepResearch Bench: Agent vs single-turn LLM paired benchmark"
    )
    parser.add_argument(
        "--suite", choices=["demo", "full"], default="demo",
        help="demo=按 seed 随机抽样，full=所选语言的全部官方题目",
    )
    parser.add_argument(
        "--sample-size", type=int, default=10,
        help="demo 无放回随机抽样题数（默认 10）",
    )
    parser.add_argument(
        "--language", choices=["zh", "en", "all"], default="zh",
        help="默认只从 50 道中文题抽样，all=全部 100 题",
    )
    parser.add_argument("--topic", type=str, default=None, help="按官方 topic 精确过滤")
    parser.add_argument(
        "--ids", type=str, default="",
        help="逗号分隔的官方数字 ID（如 1,7,42），覆盖 suite 抽样",
    )
    parser.add_argument("--repeats", type=int, default=1, help="每题成对重复次数")
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--query-file", type=str, default=None)
    parser.add_argument("--criteria-file", type=str, default=None)
    parser.add_argument("--reference-file", type=str, default=None)
    parser.add_argument("--baseline-backend", type=str, default=None)
    parser.add_argument("--baseline-temperature", type=float, default=None)
    parser.add_argument("--baseline-max-tokens", type=int, default=None)
    parser.add_argument("--judge-backend", type=str, default="mimo")
    parser.add_argument("--judge-passes", type=int, choices=[1, 2], default=2, help="2=交换 A/B 后复评")
    parser.add_argument(
        "--skip-judge", action="store_true",
        help="跳过 RACE-aligned Judge，仅保留诊断 proxy、报告和效率数据",
    )
    parser.add_argument("--dry-run", action="store_true", help="只输出选题和配置，不调用模型")
    parser.add_argument("--output-dir", type=str, default="outputs/agent_vs_llm")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")
    if args.sample_size < 1:
        parser.error("--sample-size must be >= 1")
    setup_logging(args.log_level)
    logging.getLogger("benchmark").info("Starting paired benchmark")
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()

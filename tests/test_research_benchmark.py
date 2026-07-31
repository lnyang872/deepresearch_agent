from __future__ import annotations

from types import SimpleNamespace

import pytest

from evaluation.benchmarks.research_bench import ResearchBench
from evaluation.metrics.rule_based import RuleBasedMetrics
from evaluation.metrics.stats import bootstrap_ci_paired
from scripts.run_benchmark import build_summary, select_questions
from src.core.judge import LLMJudge


def test_research_bench_has_50_questions_and_reviewed_demo() -> None:
    bench = ResearchBench()

    assert len(bench.questions) == 50
    assert len(bench.get_demo_questions()) == 10
    assert all(q["audit_status"] == "reviewed" for q in bench.get_demo_questions())
    assert all(q["reference_urls"] for q in bench.get_demo_questions())
    assert all("知识截止日期" in q["query"] for q in bench.questions)


def test_full_suite_rejects_unreviewed_questions_by_default() -> None:
    bench = ResearchBench()
    args = SimpleNamespace(
        ids="",
        suite="full",
        domain=None,
        limit=None,
        allow_unreviewed=False,
    )

    with pytest.raises(ValueError, match="not completed source audit"):
        select_questions(bench, args)

    args.allow_unreviewed = True
    assert len(select_questions(bench, args)) == 50


def test_benchmark_refuses_to_mix_results_in_existing_run_directory(
    tmp_path,
) -> None:
    from scripts.run_benchmark import run_benchmark

    run_dir = tmp_path / "existing"
    run_dir.mkdir()
    (run_dir / "records.jsonl").write_text("old result\n", encoding="utf-8")
    args = SimpleNamespace(
        config=None,
        questions_file=None,
        ids="",
        suite="demo",
        domain=None,
        limit=None,
        allow_unreviewed=False,
        run_name="existing",
        output_dir=str(tmp_path),
        repeats=1,
        seed=1,
        skip_judge=True,
        judge_backend="mimo",
        judge_passes=0,
        dry_run=True,
    )

    with pytest.raises(FileExistsError, match="not empty"):
        import asyncio

        asyncio.run(run_benchmark(args))


def test_citation_coverage_supports_evidence_ids_and_ignores_bibliography() -> None:
    report = """# Findings

Supported paragraph. [S1]

Uncited paragraph.

## References

[S2] https://example.org/source
"""

    assert RuleBasedMetrics.citation_coverage(report) == pytest.approx(0.5)


def test_composite_score_uses_canonical_factual_accuracy() -> None:
    metrics = {
        "factual_accuracy": 1.0,
        "logical_consistency": 0.0,
        "citation_coverage": 0.0,
        "bias": 0.0,
        "comprehensiveness": 0.0,
    }

    assert RuleBasedMetrics.composite_score(metrics) == pytest.approx(0.35)


def test_seeded_bootstrap_is_reproducible() -> None:
    first = bootstrap_ci_paired([0.1, 0.2, -0.1, 0.3], n_bootstrap=200, seed=7)
    second = bootstrap_ci_paired([0.1, 0.2, -0.1, 0.3], n_bootstrap=200, seed=7)

    assert first == second


def test_judge_excerpt_keeps_report_conclusion() -> None:
    report = "A" * 15000 + "\n## Conclusion\nFinal finding."
    excerpt = LLMJudge._evaluation_excerpt(report, max_chars=1000)

    assert len(excerpt) > 1000
    assert excerpt.startswith("A" * 50)
    assert "Final finding." in excerpt


def _record(question_id: str, baseline: float, agent: float) -> dict:
    def system(score: float, elapsed: float, tokens: int) -> dict:
        return {
            "status": "success",
            "elapsed_seconds": elapsed,
            "token_usage": {"total_tokens": tokens},
            "token_usage_kind": "test",
            "rule_eval": {
                "composite_score": score,
                "metrics": {dimension: score for dimension in (
                    "factual_accuracy", "citation_coverage", "comprehensiveness",
                    "logical_consistency", "bias",
                )},
            },
        }

    return {
        "question_id": question_id,
        "domain": "test",
        "baseline": system(baseline, 1.0, 100),
        "agent": system(agent, 2.0, 200),
        "judge": None,
    }


def test_summary_aggregates_repeats_at_question_level() -> None:
    records = [
        _record("q1", 0.2, 0.5),
        _record("q1", 0.4, 0.7),
        _record("q2", 0.3, 0.4),
    ]

    summary = build_summary(records, seed=11)

    assert summary["num_questions"] == 2
    assert summary["num_paired_runs"] == 3
    assert summary["rule_composite"]["mean_delta"] == pytest.approx(0.2)
    assert summary["rule_composite"]["wins"] == 2

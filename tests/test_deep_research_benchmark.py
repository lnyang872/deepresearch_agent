from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation.benchmarks.deep_research_bench import DeepResearchBench
from evaluation.benchmarks.drb_submission import validate_submission_rows, write_submission
from evaluation.metrics.rule_based import RuleBasedMetrics
from evaluation.metrics.stats import bootstrap_ci_paired
from scripts.run_benchmark import (
    build_summary,
    evaluate_with_judge,
    _cost_usd,
    _normalize_usage,
    run_agent,
    select_questions,
)
from src.core.judge import LLMJudge
from src.orchestrator.orchestrator import Orchestrator
from src.orchestrator.schemas import OrchestratorState, ResearchReport, RunConfig


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


@pytest.fixture()
def miniature_drb(tmp_path: Path) -> DeepResearchBench:
    root = tmp_path / "deep_research_bench"
    prompts = [
        {
            "id": task_id,
            "topic": "Science" if task_id % 2 else "Finance & Business",
            "language": "zh" if task_id <= 12 else "en",
            "prompt": f"研究任务 {task_id}",
        }
        for task_id in range(1, 15)
    ]
    criteria = []
    references = []
    for prompt in prompts:
        criteria.append({
            "id": prompt["id"],
            "prompt": prompt["prompt"],
            "dimension_weight": {
                "comprehensiveness": 0.4,
                "insight": 0.3,
                "instruction_following": 0.2,
                "readability": 0.1,
            },
            "criterions": {
                dimension: [{
                    "criterion": f"{dimension} criterion",
                    "explanation": "Task-specific requirement",
                    "weight": 1.0,
                }]
                for dimension in (
                    "comprehensiveness", "insight",
                    "instruction_following", "readability",
                )
            },
        })
        references.append({
            "id": prompt["id"],
            "prompt": prompt["prompt"],
            "article": f"Reference article {prompt['id']}",
        })

    _write_jsonl(root / "data/prompt_data/query.jsonl", prompts)
    _write_jsonl(root / "data/criteria_data/criteria.jsonl", criteria)
    _write_jsonl(root / "data/test_data/cleaned_data/reference.jsonl", references)
    return DeepResearchBench(dataset_root=root)


def _selection_args(**overrides) -> SimpleNamespace:
    values = {
        "ids": "",
        "suite": "demo",
        "language": "zh",
        "topic": None,
        "sample_size": 10,
        "seed": 7,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_loader_joins_official_prompt_criteria_and_reference(
    miniature_drb: DeepResearchBench,
) -> None:
    assert len(miniature_drb.questions) == 14
    first = miniature_drb.questions[0]
    assert first["id"] == "drb_001"
    assert first["source_id"] == 1
    assert first["criteria"]["dimension_weight"]["insight"] == 0.3
    assert first["reference_article"] == "Reference article 1"
    assert miniature_drb.dataset_metadata()["language_counts"] == {"zh": 12, "en": 2}


def test_drb_submission_is_normalized_to_official_schema(
    miniature_drb: DeepResearchBench, tmp_path: Path
) -> None:
    rows = [
        {
            "id": "drb_002",
            "prompt": "研究任务 2",
            "article": "Report 2",
            "status": "success",
        },
        {
            "id": 1,
            "prompt": "研究任务 1",
            "article": "Report 1",
            "metadata": {"tokens": 12},
        },
    ]
    normalized = validate_submission_rows(rows, miniature_drb)
    assert normalized == [
        {"id": 1, "prompt": "研究任务 1", "article": "Report 1"},
        {"id": 2, "prompt": "研究任务 2", "article": "Report 2"},
    ]
    path = write_submission(tmp_path / "submission.jsonl", rows, miniature_drb)
    assert path.read_text(encoding="utf-8").count("\n") == 2


def test_drb_submission_rejects_duplicate_or_mismatched_rows(
    miniature_drb: DeepResearchBench,
) -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        validate_submission_rows(
            [
                {"id": 1, "prompt": "研究任务 1", "article": "A"},
                {"id": "drb_001", "prompt": "研究任务 1", "article": "B"},
            ],
            miniature_drb,
        )
    with pytest.raises(ValueError, match="Prompt mismatch"):
        validate_submission_rows(
            [{"id": 1, "prompt": "wrong", "article": "A"}], miniature_drb
        )


def test_drb_submission_can_require_all_tasks(miniature_drb: DeepResearchBench) -> None:
    with pytest.raises(ValueError, match="Incomplete"):
        validate_submission_rows(
            [{"id": 1, "prompt": "研究任务 1", "article": "A"}],
            miniature_drb,
            require_complete=True,
        )


def test_provider_usage_and_cost_are_normalized_without_estimation() -> None:
    usage = _normalize_usage({"usage": {"prompt_tokens": 100, "completion_tokens": 25}})
    assert usage == {"prompt_tokens": 100, "completion_tokens": 25, "total_tokens": 125}
    config = {"benchmark": {"cost_per_1m_tokens": {"openai": {"input": 1.0, "output": 2.0}}}}
    assert _cost_usd(usage, config, "openai") == pytest.approx(0.00015)
    assert _cost_usd(usage, {"benchmark": {}}, "openai") is None


def test_demo_randomly_samples_ten_reproducibly(
    miniature_drb: DeepResearchBench,
) -> None:
    first = select_questions(miniature_drb, _selection_args(seed=11))
    second = select_questions(miniature_drb, _selection_args(seed=11))
    different = select_questions(miniature_drb, _selection_args(seed=12))

    assert len(first) == 10
    assert [q["source_id"] for q in first] == [q["source_id"] for q in second]
    assert [q["source_id"] for q in first] != [q["source_id"] for q in different]
    assert all(q["language"] == "zh" for q in first)


def test_explicit_official_ids_override_sampling(miniature_drb: DeepResearchBench) -> None:
    selected = select_questions(
        miniature_drb,
        _selection_args(ids="drb_014,2", language="zh", sample_size=1),
    )
    assert [q["source_id"] for q in selected] == [14, 2]


def test_benchmark_refuses_to_mix_existing_results(
    miniature_drb: DeepResearchBench, tmp_path: Path
) -> None:
    from scripts import run_benchmark as benchmark_module

    run_dir = tmp_path / "outputs" / "existing"
    run_dir.mkdir(parents=True)
    (run_dir / "records.jsonl").write_text("old result\n", encoding="utf-8")
    args = SimpleNamespace(
        config=None,
        dataset_root=str(miniature_drb.dataset_root),
        query_file=None,
        criteria_file=None,
        reference_file=None,
        ids="",
        suite="demo",
        language="zh",
        topic=None,
        sample_size=10,
        run_name="existing",
        output_dir=str(tmp_path / "outputs"),
        repeats=1,
        seed=1,
        skip_judge=True,
        judge_backend="mimo",
        judge_passes=0,
        dry_run=True,
    )

    with pytest.raises(FileExistsError, match="not empty"):
        asyncio.run(benchmark_module.run_benchmark(args))


def test_race_pairwise_weighted_overall() -> None:
    class FakeJudge:
        def compare_two_race(self, *args, **kwargs):
            return {
                "comprehensiveness": {"A": 4, "B": 2},
                "insight": {"A": 3, "B": 1},
                "instruction_following": {"A": 5, "B": 4},
                "readability": {"A": 2, "B": 5},
            }

    question = {
        "query": "task",
        "reference_article": "reference",
        "criteria": {
            "dimension_weight": {
                "comprehensiveness": 0.4,
                "insight": 0.3,
                "instruction_following": 0.2,
                "readability": 0.1,
            }
        },
    }
    result = evaluate_with_judge(
        FakeJudge(), question, {"agent": "A", "baseline": "B"}, "agent", 1
    )

    assert result["aggregate"]["overall"]["agent"] == pytest.approx(3.7)
    assert result["aggregate"]["overall"]["baseline"] == pytest.approx(2.4)


def test_citation_coverage_supports_evidence_ids_and_ignores_bibliography() -> None:
    report = """# Findings

Supported paragraph. [S1]

Uncited paragraph.

## References

[S2] https://example.org/source
"""
    assert RuleBasedMetrics.citation_coverage(report) == pytest.approx(0.5)


def test_seeded_bootstrap_is_reproducible() -> None:
    first = bootstrap_ci_paired([0.1, 0.2, -0.1, 0.3], n_bootstrap=200, seed=7)
    second = bootstrap_ci_paired([0.1, 0.2, -0.1, 0.3], n_bootstrap=200, seed=7)
    assert first == second


def test_judge_excerpt_and_compact_race_criteria_keep_required_content() -> None:
    report = "A" * 15000 + "\n## Conclusion\nFinal finding."
    excerpt = LLMJudge._evaluation_excerpt(report, max_chars=1000)
    criteria = {
        "dimension_weight": {"comprehensiveness": 1.0},
        "criterions": {
            "comprehensiveness": [{
                "criterion": "Coverage",
                "explanation": "Required details",
                "weight": 1.0,
            }]
        },
    }
    formatted = LLMJudge._format_race_criteria(criteria)

    assert excerpt.startswith("A" * 50)
    assert "Final finding." in excerpt
    assert "Coverage" in formatted


def test_judge_extracts_json_after_prose_or_markdown_fence() -> None:
    response = "Here is the evaluation:\n```json\n{\"A\": 4, \"B\": 3}\n```\n"
    assert LLMJudge._extract_json(response) == {"A": 4, "B": 3}


def test_judge_validates_race_dimensions() -> None:
    valid = {
        dimension: {"A": 4, "B": 3}
        for dimension in (
            "comprehensiveness",
            "insight",
            "instruction_following",
            "readability",
        )
    }
    assert LLMJudge._is_valid_race_result(valid)
    assert not LLMJudge._is_valid_race_result({"comprehensiveness": {"A": 4, "B": 3}})


def test_judge_passes_sampling_to_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeRouter:
        @staticmethod
        def create_backend(name: str, **kwargs: object) -> object:
            captured["name"] = name
            captured.update(kwargs)
            return object()

    monkeypatch.setattr("src.models.model_router.ModelRouter", FakeRouter)
    judge = LLMJudge("mimo", sampling={"max_tokens": 2048, "temperature": 0.1})
    judge._get_policy()

    assert captured == {"name": "mimo", "max_tokens": 2048, "temperature": 0.1}


def test_judge_reports_empty_final_content() -> None:
    class EmptyPolicy:
        def __call__(self, messages: list[dict[str, str]]) -> dict[str, object]:
            return {
                "content": "",
                "reasoning_content": "thinking",
                "usage": {"completion_tokens": 1024},
            }

    judge = LLMJudge("mimo")
    judge._policy = EmptyPolicy()
    result = judge.compare_two_race(
        "baseline", "agent", "task", {"dimension_weight": {}}, "reference"
    )

    assert "no final content" in result["error"]
    assert result["response_meta"]["reasoning_length"] == len("thinking")


def test_benchmark_marks_evidence_gap_report_as_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import run_benchmark as benchmark_module

    async def fake_run_research(*args: object, **kwargs: object) -> str:
        return "# Research Report\n\n证据不足：没有来源通过来源准入门禁，因此未保留事实性结论。"

    monkeypatch.setattr(benchmark_module, "initialize_modules", lambda *args, **kwargs: {})
    monkeypatch.setattr(benchmark_module, "run_research", fake_run_research)

    result = asyncio.run(run_agent("question", {}, "test-session"))

    assert result["status"] == "failed"
    assert result["error"] == "agent returned no admissible-evidence report"


def test_global_timeout_runs_bounded_finalization_instead_of_dropping_results() -> None:
    orchestrator = object.__new__(Orchestrator)
    orchestrator._memory_store = {}
    orchestrator._results = []
    orchestrator._dag = None
    orchestrator._task_map = {}
    orchestrator._replan_count = 0
    orchestrator._adversarial_count = 0
    orchestrator.memory_store = None
    timeout_checks = iter([False, True])
    orchestrator._is_global_timeout = lambda: next(timeout_checks)
    finalized: list[bool] = []

    async def leave_idle() -> OrchestratorState:
        return OrchestratorState.COLLECTING

    async def synthesize_partial_results() -> OrchestratorState:
        finalized.append(True)
        orchestrator._memory_store["final_report"] = ResearchReport(
            query=orchestrator._query, content="partial report"
        )
        return OrchestratorState.DONE

    orchestrator._state_handlers = {
        OrchestratorState.IDLE: leave_idle,
        OrchestratorState.SYNTHESIZING: synthesize_partial_results,
    }

    report = asyncio.run(Orchestrator.run(orchestrator, "question", RunConfig()))

    assert finalized == [True]
    assert report.content == "partial report"


def _record(question_id: str, baseline: float, agent: float) -> dict:
    def system(score: float, elapsed: float, tokens: int) -> dict:
        return {
            "status": "success",
            "elapsed_seconds": elapsed,
            "token_usage": {"total_tokens": tokens},
            "token_usage_kind": "test",
            "proxy_eval": {
                "composite_score": score,
                "metrics": {
                    dimension: score
                    for dimension in (
                        "citation_coverage",
                        "logical_consistency",
                        "low_hallucination_proxy",
                    )
                },
            },
        }

    return {
        "question_id": question_id,
        "source_id": int(question_id[1:]),
        "topic": "test",
        "language": "zh",
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
    assert summary["diagnostic_proxy_composite"]["mean_delta"] == pytest.approx(0.2)
    assert summary["diagnostic_proxy_composite"]["wins"] == 2

#!/usr/bin/env python3
"""Run the official DRB RACE and FACT pipelines for a validated submission."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.benchmarks.deep_research_bench import DeepResearchBench
from evaluation.benchmarks.drb_submission import validate_submission_file


DEFAULT_DATASET_ROOT = PROJECT_ROOT / "deep_research_bench-main"


def _run(command: list[str], *, cwd: Path) -> None:
    print("+", " ".join(str(part) for part in command))
    subprocess.run(command, cwd=str(cwd), check=True, env=os.environ.copy())


def run_official(args: argparse.Namespace) -> Path:
    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    bench = DeepResearchBench(dataset_root=dataset_root)
    required_source_ids = None
    if args.only_zh or args.only_en:
        language = "zh" if args.only_zh else "en"
        required_source_ids = {
            question["source_id"] for question in bench.get_questions(language=language)
        }
    rows = validate_submission_file(
        args.submission,
        bench,
        require_complete=not args.allow_subset and args.limit is None,
        required_source_ids=required_source_ids,
    )

    model_name = args.model_name
    if not model_name or Path(model_name).name != model_name:
        raise ValueError("--model-name must be a simple filename stem without path separators")
    race_script = dataset_root / "deepresearch_bench_race.py"
    if not race_script.is_file():
        raise FileNotFoundError(f"Official RACE script not found: {race_script}")

    staging_raw = output_dir / "staging_raw_data"
    staging_raw.mkdir(parents=True, exist_ok=True)
    staged_submission = staging_raw / f"{model_name}.jsonl"
    staged_submission.write_text(
        "".join(f"{json.dumps(row, ensure_ascii=False)}\n" for row in rows),
        encoding="utf-8",
    )
    query_file = dataset_root / "data" / "prompt_data" / "query.jsonl"
    race_output = output_dir / "race" / model_name
    cleaned_output = output_dir / "staging_cleaned_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_race:
        if args.llm_backend:
            os.environ["LLM_BACKEND"] = args.llm_backend
        if args.race_model:
            os.environ["RACE_MODEL"] = args.race_model
        race_command = [
            sys.executable,
            str(race_script),
            model_name,
            "--raw_data_dir",
            str(staging_raw),
            "--cleaned_data_dir",
            str(cleaned_output),
            "--query_file",
            str(query_file),
            "--output_dir",
            str(race_output),
            "--max_workers",
            str(args.max_workers),
        ]
        if args.limit is not None:
            race_command.extend(["--limit", str(args.limit)])
        if args.only_zh:
            race_command.append("--only_zh")
        if args.only_en:
            race_command.append("--only_en")
        if args.force:
            race_command.append("--force")
        _run(race_command, cwd=dataset_root)

    if not args.skip_fact:
        if args.llm_backend:
            os.environ["LLM_BACKEND"] = args.llm_backend
        if args.fact_model:
            os.environ["FACT_MODEL"] = args.fact_model
        fact_output = output_dir / "fact" / model_name
        fact_output.mkdir(parents=True, exist_ok=True)
        commands = [
            [
                "extract",
                "--raw_data_path",
                str(staged_submission),
                "--output_path",
                str(fact_output / "extracted.jsonl"),
                "--query_data_path",
                str(query_file),
            ],
            [
                "deduplicate",
                "--raw_data_path",
                str(fact_output / "extracted.jsonl"),
                "--output_path",
                str(fact_output / "deduplicated.jsonl"),
                "--query_data_path",
                str(query_file),
            ],
            [
                "scrape",
                "--raw_data_path",
                str(fact_output / "deduplicated.jsonl"),
                "--output_path",
                str(fact_output / "scraped.jsonl"),
            ],
            [
                "validate",
                "--raw_data_path",
                str(fact_output / "scraped.jsonl"),
                "--output_path",
                str(fact_output / "validated.jsonl"),
                "--query_data_path",
                str(query_file),
            ],
        ]
        for stage in commands:
            _run(
                [sys.executable, "-m", f"utils.{stage[0]}", *stage[1:], "--n_total_process", str(args.max_workers)],
                cwd=dataset_root,
            )
        _run(
            [
                sys.executable,
                "-m",
                "utils.stat",
                "--input_path",
                str(fact_output / "validated.jsonl"),
                "--output_path",
                str(fact_output / "fact_result.txt"),
            ],
            cwd=dataset_root,
        )
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run official DeepResearch Bench evaluation")
    parser.add_argument("--submission", required=True, help="Validated DRB JSONL with id/prompt/article")
    parser.add_argument("--model-name", required=True, help="Output model name, e.g. my_agent")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--output-dir", default="outputs/drb_official")
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only-zh", action="store_true")
    parser.add_argument("--only-en", action="store_true")
    parser.add_argument("--allow-subset", action="store_true", help="Allow a submission with fewer than all 100 tasks")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-race", action="store_true")
    parser.add_argument("--skip-fact", action="store_true")
    parser.add_argument("--race-model", default="gpt-5.5", help="Official RACE model (RACE_MODEL)")
    parser.add_argument("--fact-model", default="gpt-5.4-mini", help="Official FACT model (FACT_MODEL)")
    parser.add_argument("--llm-backend", choices=["openrouter", "openai"], default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.only_zh and args.only_en:
        raise SystemExit("--only-zh and --only-en cannot be used together")
    if args.max_workers < 1:
        raise SystemExit("--max-workers must be >= 1")
    run_official(args)


if __name__ == "__main__":
    main()

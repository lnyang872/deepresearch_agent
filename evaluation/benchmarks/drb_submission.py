"""Validation and serialization helpers for official DeepResearch Bench input."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from evaluation.benchmarks.deep_research_bench import DeepResearchBench


REQUIRED_KEYS = ("id", "prompt", "article")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid submission JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Submission row at {path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def validate_submission_rows(
    rows: Iterable[dict[str, Any]],
    bench: DeepResearchBench,
    *,
    require_complete: bool = False,
    required_source_ids: Iterable[int] | None = None,
) -> list[dict[str, Any]]:
    """Validate rows against DRB IDs and exact prompts.

    Extra input keys are accepted, but the returned rows contain only the three
    fields accepted by the official DRB raw-data format.
    """
    expected = {question["source_id"]: question["query"] for question in bench.questions}
    normalized: list[dict[str, Any]] = []
    seen: set[int] = set()

    for index, row in enumerate(rows, 1):
        missing = [key for key in REQUIRED_KEYS if key not in row]
        if missing:
            raise ValueError(f"Submission row {index} is missing required keys: {missing}")
        source_id = bench._normalize_source_id(row["id"])
        if source_id in seen:
            raise ValueError(f"Duplicate DRB submission id: {source_id}")
        if source_id not in expected:
            raise ValueError(f"Unknown DRB submission id: {row['id']!r}")
        prompt = str(row["prompt"])
        if prompt != expected[source_id]:
            raise ValueError(f"Prompt mismatch for DRB task {source_id}")
        article = str(row["article"])
        if not article.strip():
            raise ValueError(f"Empty article for DRB task {source_id}")
        seen.add(source_id)
        normalized.append({"id": source_id, "prompt": prompt, "article": article})

    if not normalized:
        raise ValueError("DRB submission is empty")
    required_ids = set(required_source_ids) if required_source_ids is not None else set(expected)
    if require_complete:
        missing_ids = sorted(required_ids - seen)
        if missing_ids:
            raise ValueError(f"Incomplete DRB submission; missing task ids: {missing_ids}")
    return sorted(normalized, key=lambda row: row["id"])


def validate_submission_file(
    path: str | Path,
    bench: DeepResearchBench,
    *,
    require_complete: bool = False,
    required_source_ids: Iterable[int] | None = None,
) -> list[dict[str, Any]]:
    return validate_submission_rows(
        _read_jsonl(Path(path)),
        bench,
        require_complete=require_complete,
        required_source_ids=required_source_ids,
    )


def write_submission(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
    bench: DeepResearchBench,
    *,
    require_complete: bool = False,
    required_source_ids: Iterable[int] | None = None,
) -> Path:
    destination = Path(path)
    normalized = validate_submission_rows(
        rows,
        bench,
        require_complete=require_complete,
        required_source_ids=required_source_ids,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in normalized:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return destination

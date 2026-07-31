"""Loader for the official DeepResearch Bench dataset checked into the workspace."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "deep_research_bench-main"


class DeepResearchBench:
    """Load prompts, task-specific RACE criteria and reference articles."""

    NAME = "DeepResearch Bench"

    def __init__(
        self,
        dataset_root: str | Path | None = None,
        query_file: str | Path | None = None,
        criteria_file: str | Path | None = None,
        reference_file: str | Path | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root or DEFAULT_DATASET_ROOT).resolve()
        self.query_file = self._resolve_path(
            query_file, "data/prompt_data/query.jsonl"
        )
        self.criteria_file = self._resolve_path(
            criteria_file, "data/criteria_data/criteria.jsonl"
        )
        self.reference_file = self._resolve_path(
            reference_file, "data/test_data/cleaned_data/reference.jsonl"
        )

        prompts = self._read_jsonl(self.query_file)
        criteria = self._index_by_id(self._read_jsonl(self.criteria_file), "criteria")
        references = self._index_by_id(self._read_jsonl(self.reference_file), "reference")

        self.questions: list[dict[str, Any]] = []
        for prompt in prompts:
            source_id = self._normalize_source_id(prompt.get("id"))
            criterion = criteria.get(source_id)
            reference = references.get(source_id)
            if criterion is None:
                raise ValueError(f"DeepResearch Bench task {source_id} has no criteria")
            if reference is None or not str(reference.get("article", "")).strip():
                raise ValueError(f"DeepResearch Bench task {source_id} has no reference article")
            self.questions.append({
                "id": f"drb_{source_id:03d}",
                "source_id": source_id,
                "topic": str(prompt.get("topic", "Unknown")),
                "domain": str(prompt.get("topic", "Unknown")),
                "language": str(prompt.get("language", "")),
                "query": str(prompt.get("prompt", "")).strip(),
                "criteria": copy.deepcopy(criterion),
                "reference_article": str(reference["article"]),
            })
        self.validate()

    def _resolve_path(
        self, supplied: str | Path | None, relative_default: str
    ) -> Path:
        path = Path(supplied) if supplied else self.dataset_root / relative_default
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"DeepResearch Bench file not found: {path}")
        return path

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"Expected JSON object at {path}:{line_number}")
                rows.append(value)
        if not rows:
            raise ValueError(f"DeepResearch Bench file is empty: {path}")
        return rows

    @classmethod
    def _index_by_id(
        cls, rows: list[dict[str, Any]], label: str
    ) -> dict[int, dict[str, Any]]:
        indexed: dict[int, dict[str, Any]] = {}
        for row in rows:
            source_id = cls._normalize_source_id(row.get("id"))
            if source_id in indexed:
                raise ValueError(f"Duplicate {label} id in DeepResearch Bench: {source_id}")
            indexed[source_id] = row
        return indexed

    @staticmethod
    def _normalize_source_id(value: Any) -> int:
        text = str(value).strip().lower()
        if text.startswith("drb_"):
            text = text[4:]
        try:
            return int(text)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid DeepResearch Bench task id: {value!r}") from exc

    def validate(self) -> None:
        ids = [question["source_id"] for question in self.questions]
        if len(ids) != len(set(ids)):
            raise ValueError("DeepResearch Bench contains duplicate task ids")
        for question in self.questions:
            if not question["query"]:
                raise ValueError(f"DeepResearch Bench task {question['source_id']} has no prompt")
            if question["language"] not in {"zh", "en"}:
                raise ValueError(
                    f"DeepResearch Bench task {question['source_id']} has invalid language"
                )
            criterion = question["criteria"]
            required = {"dimension_weight", "criterions"}
            if not required.issubset(criterion):
                raise ValueError(
                    f"DeepResearch Bench task {question['source_id']} has incomplete criteria"
                )

    def get_questions(
        self,
        language: str = "zh",
        topic: str | None = None,
    ) -> list[dict[str, Any]]:
        questions = self.questions
        if language != "all":
            questions = [q for q in questions if q["language"] == language]
        if topic:
            questions = [q for q in questions if q["topic"] == topic]
        return copy.deepcopy(questions)

    def get_by_source_ids(self, source_ids: list[int]) -> list[dict[str, Any]]:
        by_id = {question["source_id"]: question for question in self.questions}
        missing = [source_id for source_id in source_ids if source_id not in by_id]
        if missing:
            raise ValueError(f"Unknown DeepResearch Bench task ids: {missing}")
        return [copy.deepcopy(by_id[source_id]) for source_id in source_ids]

    def dataset_metadata(self) -> dict[str, Any]:
        def file_info(path: Path) -> dict[str, Any]:
            return {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }

        language_counts = {
            language: sum(q["language"] == language for q in self.questions)
            for language in ("zh", "en")
        }
        return {
            "name": self.NAME,
            "dataset_root": str(self.dataset_root),
            "num_tasks": len(self.questions),
            "language_counts": language_counts,
            "query_file": file_info(self.query_file),
            "criteria_file": file_info(self.criteria_file),
            "reference_file": file_info(self.reference_file),
        }


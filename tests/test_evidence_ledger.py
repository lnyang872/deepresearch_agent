import asyncio
import json

from src.agents.summarizer import SummarizerAgent
from src.orchestrator.schemas import AgentResult, AgentStatus, SubTask, TaskType
from src.utils.evidence_ledger import (
    build_evidence_ledger,
    enforce_inline_citations,
    render_verified_claims,
    validate_claims,
)


def _result() -> AgentResult:
    return AgentResult(
        task_id="task-a",
        status=AgentStatus.SUCCESS,
        trajectory=[{
            "role": "tool",
            "result": {
                "gate": {"status": "accepted"},
                "results": [{
                    "title": "Verified source",
                    "url": "https://example.org/source",
                    "snippet": "The source reports the tested finding.",
                    "source_quality": "high",
                    "relevance_score": 0.9,
                }]
            },
        }],
    )


def test_ledger_uses_only_admitted_trajectory_sources() -> None:
    cards = build_evidence_ledger([_result()])

    assert cards == [{
        "citation_id": "S1",
        "task_id": "task-a",
        "url": "https://example.org/source",
        "title": "Verified source",
        "evidence": "The source reports the tested finding.",
        "source_quality": "high",
        "relevance_score": 0.9,
    }]


def test_ledger_rejects_search_records_without_an_admission_decision() -> None:
    result = _result()
    del result.trajectory[0]["result"]["gate"]

    assert build_evidence_ledger([result]) == []


def test_inline_citation_gate_removes_uncited_and_unknown_claims() -> None:
    cards = build_evidence_ledger([_result()])
    content = "# Findings\n\nSupported finding. [S1]\n\nUnsupported finding.\n\nFake source. [S9]"

    cleaned, assertions = enforce_inline_citations(content, cards)

    assert "# Findings" in cleaned
    assert "Supported finding" in cleaned
    assert "Unsupported finding" not in cleaned
    assert "Fake source" not in cleaned
    assert assertions == [{"assertion": "Supported finding. [S1]", "citations": ["S1"]}]


def test_claim_validation_requires_admitted_citation_ids() -> None:
    cards = build_evidence_ledger([_result()])
    claims = validate_claims([
        {"section": "Findings", "claim": "Supported finding.", "citation_ids": ["S1"]},
        {"section": "Findings", "claim": "No evidence.", "citation_ids": []},
        {"section": "Findings", "claim": "Unknown evidence.", "citation_ids": ["S9"]},
        {"section": "Findings", "claim": "Supported finding.", "citation_ids": ["S1"]},
    ], cards)

    assert claims == [{
        "section": "Findings", "claim": "Supported finding.", "citation_ids": ["S1"]
    }]
    assert "Supported finding. [S1]" in render_verified_claims(claims)


def test_summarizer_sources_and_prompt_share_citation_ids() -> None:
    agent = SummarizerAgent(name="test", policy=object())
    result = _result()
    cards = build_evidence_ledger([result])

    claims = [{"section": "Findings", "claim": "Verified finding.", "citation_ids": ["S1"]}]
    prompt = agent._build_synthesis_prompt("question", claims, cards)
    report = agent._parse_report("question", "Finding. [S1]", [result], cards, [])

    assert "[S1] Verified source" in prompt
    assert "Verified finding. [S1]" in prompt
    assert report.sources[0]["citation_id"] == "S1"


def test_two_stage_summarizer_falls_back_to_all_verified_claims() -> None:
    class Policy:
        tools = None
        response_format = None
        guided_json = None

        def __call__(self, messages):
            if self.response_format:
                return {"content": json.dumps({"claims": [
                    {"section": "Findings", "claim": "First verified claim.", "citation_ids": ["S1"]},
                    {"section": "Findings", "claim": "Second verified claim.", "citation_ids": ["S1"]},
                ]})}
            return {"content": "## Findings\n\nFirst verified claim. [S1]"}

    result = asyncio.run(SummarizerAgent(name="test", policy=Policy()).run(
        SubTask(task_id="synthesize", task_type=TaskType.ANALYZE, description="summary"),
        {"query": "question", "results": [_result()]},
    ))
    report = result.output

    assert "First verified claim. [S1]" in report.content
    assert "Second verified claim. [S1]" in report.content
    assert any(step.get("stage") == "fallback" for step in result.trajectory)

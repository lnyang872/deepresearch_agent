from src.agents.summarizer import SummarizerAgent
from src.orchestrator.schemas import AgentResult, AgentStatus
from src.utils.evidence_ledger import build_evidence_ledger, enforce_inline_citations


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


def test_summarizer_sources_and_prompt_share_citation_ids() -> None:
    agent = SummarizerAgent(name="test", policy=object())
    result = _result()
    cards = build_evidence_ledger([result])

    prompt = agent._build_synthesis_prompt("question", [result], cards)
    report = agent._parse_report("question", "Finding. [S1]", [result], cards, [])

    assert "[S1] Verified source" in prompt
    assert report.sources[0]["citation_id"] == "S1"

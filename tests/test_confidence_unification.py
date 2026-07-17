from src.agents.summarizer import SummarizerAgent
from src.core.runner import _format_report
from src.orchestrator.schemas import AgentResult, AgentStatus, ResearchReport
from src.utils.report_content import strip_embedded_overall_confidence


def test_strips_chinese_and_english_llm_confidence_lines() -> None:
    content = """# Report

Finding.
## 总体置信度: 0.46
**总置信度0.72**的原因：模型自行解释。
Overall Confidence: 0.91

Conclusion.
"""

    cleaned = strip_embedded_overall_confidence(content)

    assert cleaned == "# Report\n\nFinding.\n\nConclusion."


def test_report_confidence_is_not_affected_by_llm_self_score() -> None:
    agent = SummarizerAgent(name="test", policy=object())
    results = [
        AgentResult(task_id="a", status=AgentStatus.SUCCESS),
        AgentResult(task_id="b", status=AgentStatus.SUCCESS),
        AgentResult(task_id="c", status=AgentStatus.FAILED),
    ]

    report = agent._parse_report(
        "query", "Overall Confidence: 1.00\n正文", results
    )

    assert report.confidence == 0.67


def test_final_output_has_one_application_owned_confidence() -> None:
    report = ResearchReport(
        query="query",
        content="正文\n\n总体置信度: 0.12\n\n尾声",
        confidence=0.67,
    )

    rendered = _format_report(report, elapsed=1.0)

    assert "总体置信度" not in rendered
    assert rendered.count("**置信度**: 0.67") == 1


def test_final_output_rejects_uncited_content_after_late_rewrite() -> None:
    report = ResearchReport(
        query="query",
        content="# Findings\n\nSupported statement. [S1]\n\nInvented statement.",
        sources=[{
            "citation_id": "S1",
            "title": "Source",
            "url": "https://example.org/source",
            "snippet": "Supporting excerpt.",
        }],
    )

    rendered = _format_report(report, elapsed=1.0)

    assert "Supported statement" in rendered
    assert "Invented statement" not in rendered
    assert report.evidence_ledger == [{
        "assertion": "Supported statement. [S1]", "citations": ["S1"]
    }]


def test_final_output_lists_only_sources_used_by_inline_citations() -> None:
    report = ResearchReport(
        query="query",
        content="Supported statement. [S1]",
        sources=[
            {"citation_id": "S1", "title": "Used", "url": "https://example.org/used"},
            {"citation_id": "S2", "title": "Unused", "url": "https://example.org/unused"},
        ],
    )

    rendered = _format_report(report, elapsed=1.0)

    assert "[S1] [Used]" in rendered
    assert "[S2] [Unused]" not in rendered

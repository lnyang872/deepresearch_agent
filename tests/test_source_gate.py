import asyncio
import json

from src.agents.researcher import ResearcherAgent
from src.utils.source_gate import SourceGate


def test_web_gate_keeps_relevant_high_quality_result_only() -> None:
    response = {
        "results": [
            {
                "title": "Gender Bias in Coreference Resolution",
                "url": "https://aclanthology.org/N18-2003.pdf?ref=search",
                "snippet": "This paper evaluates gender bias in coreference resolution systems.",
            },
            {
                "title": "Coreference Resolution Notes",
                "url": "https://www.docin.com/p-868794941.html",
                "snippet": "Coreference resolution tutorial.",
            },
            {
                "title": "Geological disaster response knowledge graph",
                "url": "https://example.org/geology",
                "snippet": "A study of emergency response and geologic hazards.",
            },
            {
                "title": "Duplicate result",
                "url": "https://aclanthology.org/N18-2003.pdf",
                "snippet": "Coreference resolution duplicate.",
            },
        ]
    }

    filtered = SourceGate().filter_web_response(
        "large language model pronoun coreference resolution", response
    )

    assert filtered["total"] == 1
    assert filtered["results"][0]["source_quality"] == "high"
    assert filtered["results"][0]["url"] == "https://aclanthology.org/N18-2003.pdf"
    assert filtered["gate"]["rejection_reasons"] == {
        "low_quality_domain": 1,
        "irrelevant": 1,
        "duplicate": 1,
    }


def test_paper_gate_rejects_irrelevant_papers_and_keeps_metadata() -> None:
    response = {
        "papers": [
            {
                "title": "Coreference Resolution with Pretrained Language Models",
                "summary": "A study of language-model approaches to coreference resolution.",
                "pdf_url": "https://arxiv.org/abs/2504.05855",
            },
            {
                "title": "Image Super Resolution with Multi-Attention Fusion",
                "summary": "A computer vision method for single image super resolution.",
                "pdf_url": "https://arxiv.org/abs/2401.00001",
            },
        ]
    }

    filtered = SourceGate().filter_paper_response("language model coreference resolution", response)

    assert [paper["title"] for paper in filtered["papers"]] == [
        "Coreference Resolution with Pretrained Language Models"
    ]
    assert filtered["gate"]["rejection_reasons"] == {"irrelevant": 1}


def test_gate_returns_empty_result_with_diagnostic_when_nothing_qualifies() -> None:
    response = {
        "results": [
            {
                "title": "Mathematics homework design",
                "url": "https://example.org/math",
                "snippet": "A study of middle-school homework.",
            }
        ]
    }

    filtered = SourceGate().filter_web_response("large language model coreference", response)

    assert filtered["results"] == []
    assert filtered["gate"]["status"] == "no_eligible_sources"
    assert filtered["gate"]["rejection_reasons"] == {"irrelevant": 1}


def test_researcher_trajectory_contains_only_admitted_results() -> None:
    class SearchTool:
        name = "web_search"

        async def execute(self, query: str) -> dict:
            return {
                "results": [
                    {
                        "title": "Coreference Resolution with Language Models",
                        "url": "https://aclanthology.org/2025.coref.pdf",
                        "snippet": "Language model evaluation for coreference resolution.",
                    },
                    {
                        "title": "Homework design",
                        "url": "https://example.org/homework",
                        "snippet": "Middle-school mathematics homework research.",
                    },
                ]
            }

    agent = ResearcherAgent(name="test", policy=object(), tools=[SearchTool()])
    trajectory: list[dict] = []
    tool_calls = [{
        "id": "search-1",
        "function": {
            "name": "web_search",
            "arguments": json.dumps({"query": "language model coreference resolution"}),
        },
    }]

    tool_results, errors = asyncio.run(
        agent._execute_tools_parallel(tool_calls, trajectory, 0, "fallback query")
    )

    assert errors == []
    assert tool_results[0]["result"]["total"] == 1
    assert "Homework design" not in str(trajectory)

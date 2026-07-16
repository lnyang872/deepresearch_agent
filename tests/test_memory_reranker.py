from src.memory.long_term import MemoryEntry
from src.memory.reranker import MemoryReranker


def _entry(entry_id: str, claim: str, topic: str, source: str) -> MemoryEntry:
    return MemoryEntry(
        entry_id=entry_id,
        claim=claim,
        source=source,
        confidence=0.9,
        agent_id="test",
        timestamp=0.0,
        evidence_type="primary",
        embedding=[],
        topic=topic,
    )


def test_memory_reranker_promotes_claim_matching_query_topic():
    reranker = MemoryReranker()
    query = "DeepSeek R1 GRPO technical report"

    candidates = [
        (
            _entry(
                "a",
                "This paper studies retrieval and citation quality in academic QA systems.",
                "academic qa",
                "paper-a",
            ),
            0.82,
        ),
        (
            _entry(
                "b",
                "DeepSeek-R1 introduces GRPO for reasoning-oriented post-training.",
                "deepseek r1 grpo",
                "paper-b",
            ),
            0.75,
        ),
    ]

    reranked = reranker.rerank(query, candidates, top_k=2)
    assert reranked[0][0].entry_id == "b"


def test_memory_reranker_can_disable_cross_encoder_and_use_heuristic_path():
    reranker = MemoryReranker(use_cross_encoder=False)
    query = "DeepSeek R1 GRPO technical report"

    candidates = [
        (
            _entry(
                "a",
                "This paper studies retrieval and citation quality in academic QA systems.",
                "academic qa",
                "paper-a",
            ),
            0.82,
        ),
        (
            _entry(
                "b",
                "DeepSeek-R1 introduces GRPO for reasoning-oriented post-training.",
                "deepseek r1 grpo",
                "paper-b",
            ),
            0.75,
        ),
    ]

    reranked = reranker.rerank(query, candidates, top_k=2)
    assert reranked[0][0].entry_id == "b"

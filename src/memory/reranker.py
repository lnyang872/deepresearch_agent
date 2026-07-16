"""
Memory reranker for second-stage ranking of retrieved evidence.

This module keeps the project inference-only: it does not require a
cross-encoder by default or extra training artifacts. It reranks recalled
memory entries by combining:
  1. first-stage retrieval score
  2. semantic similarity between query and claim/topic text
  3. lexical overlap between query and claim/topic/source
  4. a small topic match bonus
"""

from __future__ import annotations

import logging
import re
from typing import Iterable

import numpy as np

from .embedder import Embedder
from .long_term import MemoryEntry

__all__ = ["MemoryReranker"]

logger = logging.getLogger(__name__)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _tokenize(text: str) -> list[str]:
    """Tokenize mixed Chinese/English text with a simple, dependency-free rule."""
    if not text:
        return []
    text = text.lower()
    english = re.findall(r"[a-z0-9_]+", text)
    chinese = re.findall(r"[\u4e00-\u9fff]", text)
    return english + chinese


def _jaccard_overlap(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    inter = len(left_set & right_set)
    union = len(left_set | right_set)
    return inter / union if union else 0.0


class MemoryReranker:
    """Second-stage reranker for memory retrieval candidates."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        retrieval_weight: float = 0.20,
        semantic_weight: float = 0.50,
        lexical_weight: float = 0.20,
        topic_weight: float = 0.10,
        use_cross_encoder: bool = False,
        cross_encoder_model_name: str = "BAAI/bge-reranker-v2-m3",
    ) -> None:
        self.embedder = embedder or Embedder()
        self.retrieval_weight = retrieval_weight
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight
        self.topic_weight = topic_weight
        self.use_cross_encoder = use_cross_encoder
        self.cross_encoder_model_name = cross_encoder_model_name
        self._cross_encoder = None
        self._cross_encoder_available = False

    def _load_cross_encoder(self):
        """Lazy-load a cross-encoder reranker when enabled."""
        if not self.use_cross_encoder:
            return None
        if self._cross_encoder is not None:
            return self._cross_encoder
        try:
            from sentence_transformers import CrossEncoder

            self._cross_encoder = CrossEncoder(self.cross_encoder_model_name)
            self._cross_encoder_available = True
            logger.info(
                "Loaded cross-encoder reranker: %s", self.cross_encoder_model_name
            )
        except Exception as exc:
            self._cross_encoder = None
            self._cross_encoder_available = False
            logger.warning(
                "Failed to load cross-encoder reranker '%s', falling back to heuristic reranker: %s",
                self.cross_encoder_model_name,
                exc,
            )
        return self._cross_encoder

    def rerank(
        self,
        query: str,
        candidates: list[tuple[MemoryEntry, float]],
        top_k: int | None = None,
    ) -> list[tuple[MemoryEntry, float]]:
        """
        Rerank candidates recalled by first-stage retrieval.

        Args:
            query: user query
            candidates: [(entry, first_stage_score), ...]
            top_k: optional truncation after reranking

        Returns:
            [(entry, reranked_score), ...] sorted by reranked_score desc
        """
        if not query or not candidates:
            return candidates[:top_k] if top_k is not None else candidates

        cross_encoder = self._load_cross_encoder()
        if cross_encoder is not None:
            return self._rerank_with_cross_encoder(query, candidates, top_k)

        query_vec = np.array(self.embedder.encode(query), dtype=np.float32)
        query_tokens = _tokenize(query)
        reranked: list[tuple[MemoryEntry, float]] = []

        for entry, retrieval_score in candidates:
            claim_vec = np.array(
                entry.embedding if entry.embedding else self.embedder.encode(entry.claim),
                dtype=np.float32,
            )
            semantic = _cosine(query_vec, claim_vec)

            evidence_text = " ".join(
                part for part in [entry.claim, entry.topic, entry.source] if part
            )
            lexical = _jaccard_overlap(query_tokens, _tokenize(evidence_text))
            topic_overlap = _jaccard_overlap(query_tokens, _tokenize(entry.topic))

            final_score = (
                retrieval_score * self.retrieval_weight
                + semantic * self.semantic_weight
                + lexical * self.lexical_weight
                + topic_overlap * self.topic_weight
            )
            reranked.append((entry, final_score))

        reranked.sort(key=lambda item: item[1], reverse=True)
        return reranked[:top_k] if top_k is not None else reranked

    def _rerank_with_cross_encoder(
        self,
        query: str,
        candidates: list[tuple[MemoryEntry, float]],
        top_k: int | None = None,
    ) -> list[tuple[MemoryEntry, float]]:
        """Rerank candidates using a learned cross-encoder model."""
        assert self._cross_encoder is not None

        pairs = []
        for entry, _ in candidates:
            evidence_text = " [SEP] ".join(
                part for part in [entry.topic, entry.claim, entry.source] if part
            )
            pairs.append((query, evidence_text))

        try:
            ce_scores = self._cross_encoder.predict(pairs)
        except Exception as exc:
            logger.warning(
                "Cross-encoder scoring failed, falling back to heuristic reranker: %s",
                exc,
            )
            self._cross_encoder = None
            self._cross_encoder_available = False
            return self.rerank(query, candidates, top_k)

        reranked = [
            (entry, float(score))
            for (entry, _), score in zip(candidates, ce_scores)
        ]
        reranked.sort(key=lambda item: item[1], reverse=True)
        return reranked[:top_k] if top_k is not None else reranked

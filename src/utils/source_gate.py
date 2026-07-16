"""Deterministic admission control for search results used in research reports."""
from __future__ import annotations

from collections import Counter
import re
from typing import Any
from urllib.parse import urlparse

__all__ = ["SourceGate"]


class SourceGate:
    """Keep only relevant, non-duplicate, non-aggregator search results.

    This is intentionally deterministic: an LLM never decides whether a search
    hit is eligible evidence. Search snippets are still discovery material, not
    proof for a claim; full-text verification is handled by the browser stage.
    """

    max_results = 5
    min_relevance = 0.20

    _LOW_QUALITY_DOMAINS = {
        "baidu.com", "book118.com", "cnblogs.com", "csdn.net", "devpress.com",
        "dict.cn", "doc88.com", "docin.com", "ichacha.net", "juejin.cn",
        "researchgate.net", "sohu.com", "sina.com", "toutiao.com", "xjishu.com",
        "jigao616.com", "zhihu.com", "163.com",
    }
    _HIGH_QUALITY_DOMAINS = {
        "aclanthology.org", "ai.meta.com", "anthropic.com", "arxiv.org",
        "dl.acm.org", "ieee.org", "link.springer.com", "microsoft.com", "nature.com",
        "openai.com", "openreview.net", "proceedings.neurips.cc", "science.org",
        "semanticscholar.org", "who.int", "worldbank.org", "oecd.org", "nist.gov",
    }
    _STOPWORDS = {
        "about", "analysis", "and", "are", "based", "for", "from", "how", "in",
        "into", "latest", "of", "on", "or", "overview", "research", "study", "the",
        "to", "what", "with", "以及", "什么", "关于", "分析", "哪些", "如何", "报告",
        "技术", "方向", "最新", "研究", "进展", "需要", "问题",
    }

    def filter_web_response(self, query: str, response: dict[str, Any]) -> dict[str, Any]:
        """Filter a web-search response and return only admissible results."""
        return self._filter_response(
            query=query,
            response=response,
            collection_key="results",
            url_key="url",
            text_keys=("title", "snippet"),
        )

    def filter_paper_response(self, query: str, response: dict[str, Any]) -> dict[str, Any]:
        """Filter paper-search results by relevance and canonical paper URL."""
        return self._filter_response(
            query=query,
            response=response,
            collection_key="papers",
            url_key="pdf_url",
            text_keys=("title", "summary"),
        )

    def _filter_response(
        self,
        query: str,
        response: dict[str, Any],
        collection_key: str,
        url_key: str,
        text_keys: tuple[str, ...],
    ) -> dict[str, Any]:
        raw_items = response.get(collection_key, []) if isinstance(response, dict) else []
        accepted: list[dict[str, Any]] = []
        rejected = Counter()
        seen_urls: set[str] = set()

        for item in raw_items:
            if not isinstance(item, dict):
                rejected["invalid_record"] += 1
                continue

            url = str(item.get(url_key, "")).strip()
            canonical_url, domain = self._canonical_url(url)
            if not canonical_url:
                rejected["invalid_url"] += 1
                continue
            if domain in self._LOW_QUALITY_DOMAINS or any(
                domain.endswith(f".{blocked}") for blocked in self._LOW_QUALITY_DOMAINS
            ):
                rejected["low_quality_domain"] += 1
                continue
            if canonical_url in seen_urls:
                rejected["duplicate"] += 1
                continue

            text = " ".join(str(item.get(key, "")) for key in text_keys).strip()
            if not text:
                rejected["missing_metadata"] += 1
                continue

            relevance = self._relevance(query, text, str(item.get("title", "")))
            if relevance < self.min_relevance:
                rejected["irrelevant"] += 1
                continue

            seen_urls.add(canonical_url)
            accepted_item = dict(item)
            accepted_item[url_key] = canonical_url
            accepted_item["source_quality"] = self._source_quality(domain)
            accepted_item["relevance_score"] = round(relevance, 3)
            accepted.append(accepted_item)

        accepted.sort(
            key=lambda item: (item["source_quality"] == "high", item["relevance_score"]),
            reverse=True,
        )
        accepted = accepted[: self.max_results]

        filtered = dict(response)
        filtered[collection_key] = accepted
        if collection_key == "results":
            filtered["total"] = len(accepted)
        filtered["gate"] = {
            "status": "accepted" if accepted else "no_eligible_sources",
            "query": query,
            "received": len(raw_items),
            "accepted": len(accepted),
            "rejected": sum(rejected.values()),
            "rejection_reasons": dict(rejected),
        }
        return filtered

    @classmethod
    def _canonical_url(cls, url: str) -> tuple[str, str]:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return "", ""
        domain = parsed.netloc.lower().split(":", 1)[0]
        if domain.startswith("www."):
            domain = domain[4:]
        path = parsed.path.rstrip("/")
        return f"{parsed.scheme}://{domain}{path}", domain

    @classmethod
    def _source_quality(cls, domain: str) -> str:
        if (
            domain in cls._HIGH_QUALITY_DOMAINS
            or any(domain.endswith(f".{trusted}") for trusted in cls._HIGH_QUALITY_DOMAINS)
            or domain.endswith((".gov", ".edu", ".ac.uk", ".ac.cn"))
        ):
            return "high"
        return "standard"

    @classmethod
    def _relevance(cls, query: str, text: str, title: str) -> float:
        query_terms = cls._terms(query)
        candidate_terms = cls._terms(text)
        if not query_terms or not candidate_terms:
            return 0.0

        overlap = query_terms & candidate_terms
        min_overlap = 1 if len(query_terms) <= 3 else 2
        if len(overlap) < min_overlap:
            return 0.0
        lexical = len(overlap) / min(len(query_terms), 8)
        title_terms = cls._terms(title)
        title_overlap = len(query_terms & title_terms) / min(len(query_terms), 8)
        char_similarity = cls._char_bigram_jaccard(query, text)
        return 0.65 * lexical + 0.25 * title_overlap + 0.10 * char_similarity

    @classmethod
    def _terms(cls, text: str) -> set[str]:
        terms = {
            token.lower()
            for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text)
            if token.lower() not in cls._STOPWORDS
        }
        for segment in re.findall(r"[\u4e00-\u9fff]{2,}", text):
            terms.update(segment[index:index + 2] for index in range(len(segment) - 1))
        return terms

    @staticmethod
    def _char_bigram_jaccard(left: str, right: str) -> float:
        def grams(value: str) -> set[str]:
            compact = re.sub(r"\s+", "", value.lower())
            return {compact[index:index + 2] for index in range(len(compact) - 1)}

        left_grams, right_grams = grams(left), grams(right)
        if not left_grams or not right_grams:
            return 0.0
        return len(left_grams & right_grams) / len(left_grams | right_grams)

"""Semantic accuracy: similarity between the (system_prompt + prompt) text
and the agent output.

Default backend is TF-IDF cosine similarity — lightweight and
dependency-free, but still lexical rather than semantic (see
docs/methodology.md, Threats to Validity): two outputs that are
semantically equivalent but lexically different (paraphrases, different
variable names) score as dissimilar. The corpus for IDF is just the two
documents being compared (prompt vs output), a known simplification for
single-pair comparison absent a larger reference corpus.

An optional embedding backend closes that gap without forcing a heavy
dependency on everyone: pass ``embedding_fn`` (any callable mapping text to
a dense vector — e.g. a sentence-transformers model, an OpenAI/Anthropic
embeddings API call, a cached lookup) and cosine similarity is computed in
that embedding space instead. Document the embedding model/version
alongside any results that use it — same reproducibility expectation as
agent CLI versions (see docs/methodology.md Section 7).
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Callable, Sequence

from .base import Metric, MetricResult

_TOKEN_RE = re.compile(r"\b[a-z]{2,}\b")
_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "and", "or", "but", "if", "then", "so", "yet", "nor", "not", "no",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
    "them", "my", "your", "his", "its", "our", "their", "this", "that",
    "these", "those", "what", "how", "when", "where", "why", "which", "who",
}


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP_WORDS]


def _tfidf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    tf = Counter(tokens)
    total = sum(tf.values()) or 1
    return {term: (count / total) * idf.get(term, 0.0) for term, count in tf.items()}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    shared = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in shared)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _dense_cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"embedding_fn returned mismatched dimensions: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticAccuracyMetric(Metric):
    """Similarity between input (system_prompt+prompt) and output.

    Uses TF-IDF cosine similarity by default; supply ``embedding_fn`` to
    switch to embedding-space cosine similarity instead (see module
    docstring). Embedding cosine similarity can be negative, so its score
    is mapped from [-1, 1] to [0, 100]; TF-IDF weights are non-negative, so
    its score is simply similarity * 100.
    """

    name = "semantic_accuracy"

    def __init__(self, embedding_fn: Callable[[str], Sequence[float]] | None = None) -> None:
        self.embedding_fn = embedding_fn

    def compute(self, *, output: str, prompt: str, system_prompt: str = "", **context: Any) -> MetricResult:
        input_text = f"{system_prompt} {prompt}".strip()
        if not input_text:
            return MetricResult(score=50.0, details={"note": "no input text to compare; neutral score"})
        if not output.strip():
            return MetricResult(score=0.0, details={"cosine_similarity": 0.0})

        if self.embedding_fn is not None:
            return self._compute_embedding(input_text, output)
        return self._compute_tfidf(input_text, output)

    def _compute_embedding(self, input_text: str, output: str) -> MetricResult:
        vec_input = self.embedding_fn(input_text)
        vec_output = self.embedding_fn(output)
        similarity = _dense_cosine(vec_input, vec_output)  # in [-1, 1]
        score = max(0.0, min((similarity + 1.0) / 2.0 * 100.0, 100.0))
        return MetricResult(
            score=score,
            details={"backend": "embedding", "cosine_similarity": similarity, "dims": len(vec_input)},
        )

    def _compute_tfidf(self, input_text: str, output: str) -> MetricResult:
        input_tokens = _tokenize(input_text)
        output_tokens = _tokenize(output)

        if not input_tokens:
            return MetricResult(score=50.0, details={"note": "no keywords to compare; neutral score"})
        if not output_tokens:
            return MetricResult(score=0.0, details={"backend": "tfidf", "cosine_similarity": 0.0})

        doc_freq: Counter[str] = Counter()
        for term in set(input_tokens):
            doc_freq[term] += 1
        for term in set(output_tokens):
            doc_freq[term] += 1
        # 2-document IDF: log(N/df) + 1, N=2, smoothed to avoid zero for shared terms
        idf = {term: math.log(2 / df) + 1.0 for term, df in doc_freq.items()}

        vec_input = _tfidf_vector(input_tokens, idf)
        vec_output = _tfidf_vector(output_tokens, idf)
        similarity = _cosine(vec_input, vec_output)  # in [0, 1] for non-negative TF-IDF weights

        score = max(0.0, min(similarity * 100.0, 100.0))
        return MetricResult(
            score=score,
            details={
                "backend": "tfidf",
                "cosine_similarity": similarity,
                "input_token_count": len(set(input_tokens)),
                "output_token_count": len(set(output_tokens)),
            },
        )

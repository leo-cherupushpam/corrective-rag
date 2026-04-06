"""
reranker.py
===========
Document reranking using BGE (local, fast, cost-free).

Reranking improves CRAG precision by filtering irrelevant documents BEFORE grading.

Flow:
  1. Retrieve top-k documents (k=10 to give reranker more candidates)
  2. Rerank with BGE → get top 3-5 by semantic relevance
  3. Grade only the reranked documents (cost savings: ~20-30% fewer grade calls)

Design decisions:
- BGE (BAAI-bge-reranker-base): Free, local, ~8ms per call, trained on retrieval
- Lazy load model on first call (cache at module level)
- Batch scoring (embed all docs once, compute similarities)
- Should only rerank if we have 5+ docs to filter
"""

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Global model cache (lazy-loaded)
_MODEL = None


@dataclass
class RerankerResult:
    """Result of reranking a single document."""
    document: str          # Original document text
    score: float          # Relevance score (0.0–1.0, higher = more relevant)
    rank: int             # Rank position (0 = highest score)
    preview: str          # First 150 chars of document


def _get_model():
    """Lazy-load BGE reranker model (cached at module level)."""
    global _MODEL
    if _MODEL is None:
        try:
            from sentence_transformers import CrossEncoder
            _MODEL = CrossEncoder("BAAI/bge-reranker-base", max_length=512)
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for reranking. "
                "Install with: pip install sentence-transformers"
            )
    return _MODEL


def should_rerank(documents: list[str]) -> bool:
    """
    Return True if we should rerank these documents.
    Only rerank if we have enough docs to filter meaningfully.
    """
    return len(documents) >= 5


def score_documents(
    query: str,
    documents: list[str],
    k: int = 3,
) -> list[RerankerResult]:
    """
    Rerank documents by relevance to the query using BGE.

    Args:
        query: The user's question/search query
        documents: List of retrieved documents to rerank
        k: How many top documents to return

    Returns:
        List of top-k RerankerResult objects, ordered by score (highest first)
    """
    if not documents:
        return []

    model = _get_model()
    k = min(k, len(documents))  # Don't return more than we have

    # Prepare query-document pairs for scoring
    # Format: [(query, doc1), (query, doc2), ...]
    pairs = [[query, doc] for doc in documents]

    # Score all pairs (returns array of shape [len(pairs)])
    scores = model.predict(pairs)

    # Create result objects with scores
    results = []
    for doc, score in zip(documents, scores):
        results.append({
            "document": doc,
            "score": float(score),
            "preview": doc[:150].replace("\n", " ") + ("…" if len(doc) > 150 else ""),
        })

    # Sort by score (highest first)
    results.sort(key=lambda x: x["score"], reverse=True)

    # Create RerankerResult objects with rank
    reranked = [
        RerankerResult(
            document=r["document"],
            score=r["score"],
            rank=i,
            preview=r["preview"],
        )
        for i, r in enumerate(results[:k])
    ]

    return reranked


def rerank_documents(
    query: str,
    documents: list[str],
    k: int = 3,
) -> list[str]:
    """
    Rerank documents and return only the top-k document texts.

    Convenience wrapper around score_documents() that returns just the text.

    Args:
        query: The user's question
        documents: Retrieved documents to rerank
        k: Number of top documents to return

    Returns:
        List of top-k document strings in reranked order
    """
    if not should_rerank(documents):
        # If too few docs, don't bother reranking
        return documents

    results = score_documents(query, documents, k=k)
    return [r.document for r in results]


def get_reranking_info(results: list[RerankerResult]) -> dict:
    """
    Get human-readable info about reranking results.

    Returns:
        Dict with summary stats: top_score, avg_score, min_score, docs_count
    """
    if not results:
        return {
            "top_score": 0.0,
            "avg_score": 0.0,
            "min_score": 0.0,
            "docs_count": 0,
        }

    scores = [r.score for r in results]
    return {
        "top_score": max(scores),
        "avg_score": sum(scores) / len(scores),
        "min_score": min(scores),
        "docs_count": len(results),
    }

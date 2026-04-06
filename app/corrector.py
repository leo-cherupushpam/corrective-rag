"""
corrector.py
============
Query correction strategies when grader rejects retrieved documents.

Strategy 1 — Query expansion:  Rewrite query to be broader/more specific
Strategy 2 — Decomposition:    Break complex multi-part queries into sub-queries
Strategy 3 — Keyword fallback: Extract bare keywords for BM25-style matching

Design decisions:
- Strategies are tried in order; stop when one yields passing grade
- Max 2 correction attempts to avoid infinite loops (cost + latency)
- Each strategy produces a list of reformulated queries to try
- v1.6: Strategy registry — register custom domain-specific strategies
  Usage: register_strategy("medical_synonyms", my_fn)
         CORRECTION_STRATEGIES = ["medical_synonyms", "expand", "keywords"]
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CORRECTOR_MODEL = "gpt-4o-mini-2024-07-18"  # v1.5: consistent with grader model


class ExpandedQuery(BaseModel):
    expanded: str        # Broader/more specific reformulation
    rationale: str       # Why this expansion might help


class DecomposedQueries(BaseModel):
    sub_queries: list[str]   # Each sub-query is simpler and focused
    rationale: str


class KeywordQuery(BaseModel):
    keywords: list[str]      # Core nouns/verbs for keyword matching
    boolean_query: str       # e.g. "return policy AND refund AND days"


def expand_query(query: str) -> ExpandedQuery:
    """
    Strategy 1: Rewrite the query to be broader or more specific.
    Used when: query is too narrow/jargon-heavy and misses relevant docs.
    """
    response = client.beta.chat.completions.parse(
        model=CORRECTOR_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a search query optimizer. "
                    "Rewrite the query to be clearer, broader, or use different terminology "
                    "that might match documents better. Do not change the intent."
                ),
            },
            {"role": "user", "content": f"Original query: {query}\n\nRewrite it to improve document retrieval."},
        ],
        response_format=ExpandedQuery,
    )
    return response.choices[0].message.parsed


def decompose_query(query: str) -> DecomposedQueries:
    """
    Strategy 2: Break complex multi-part query into focused sub-queries.
    Used when: query asks multiple things; retriever can't satisfy all at once.
    """
    response = client.beta.chat.completions.parse(
        model=CORRECTOR_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a query decomposer. "
                    "Break the user's query into 2–3 simple, focused sub-questions. "
                    "Each sub-question should be answerable independently."
                ),
            },
            {"role": "user", "content": f"Query to decompose: {query}"},
        ],
        response_format=DecomposedQueries,
    )
    return response.choices[0].message.parsed


def extract_keywords(query: str) -> KeywordQuery:
    """
    Strategy 3: Extract core keywords for sparse/BM25-style retrieval.
    Used when: dense embedding retrieval fails; sparse search may catch exact terms.
    """
    response = client.beta.chat.completions.parse(
        model=CORRECTOR_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract the most important keywords from this query for document retrieval. "
                    "Focus on nouns, verbs, and domain terms. Ignore filler words. "
                    "Also write a boolean search query using AND/OR operators."
                ),
            },
            {"role": "user", "content": f"Query: {query}"},
        ],
        response_format=KeywordQuery,
    )
    return response.choices[0].message.parsed


def get_correction_candidates(query: str, strategy: str = "expand") -> list[str]:
    """
    Return a list of reformulated query strings to try.

    Args:
        query: Original query that failed grading
        strategy: Any registered strategy name ("expand" | "decompose" | "keywords"
                  or a custom-registered strategy)

    Returns:
        List of query strings to re-retrieve with
    """
    # v1.6: Look up from registry (supports custom strategies)
    if strategy in _STRATEGY_REGISTRY:
        return _STRATEGY_REGISTRY[strategy](query)

    return [query]  # No-op fallback for unknown strategies


CORRECTION_STRATEGIES = ["expand", "decompose", "keywords"]


# ---------------------------------------------------------------------------
# v1.6: Strategy registry — pluggable custom correction strategies
# ---------------------------------------------------------------------------

# Registry maps strategy name → callable(query: str) -> list[str]
_STRATEGY_REGISTRY: dict[str, callable] = {
    "expand": lambda q: [expand_query(q).expanded],
    "decompose": lambda q: decompose_query(q).sub_queries,
    "keywords": lambda q: [extract_keywords(q).boolean_query] + extract_keywords(q).keywords[:3],
}


def register_strategy(name: str, fn: callable) -> None:
    """
    Register a custom correction strategy.

    Args:
        name: Unique strategy name (e.g., "medical_synonyms")
        fn:   Callable(query: str) -> list[str]
              Must return a list of reformulated query strings to try.

    Example:
        def expand_medical(query: str) -> list[str]:
            # Add medical synonyms, ICD codes, etc.
            return [f"{query} symptoms treatment diagnosis"]

        register_strategy("medical_synonyms", expand_medical)

        # Then use it in your pipeline:
        CORRECTION_STRATEGIES = ["medical_synonyms", "expand", "keywords"]
    """
    _STRATEGY_REGISTRY[name] = fn


def list_strategies() -> list[str]:
    """Return all registered strategy names (built-in + custom)."""
    return list(_STRATEGY_REGISTRY.keys())

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
        strategy: "expand" | "decompose" | "keywords"

    Returns:
        List of query strings to re-retrieve with
    """
    if strategy == "expand":
        result = expand_query(query)
        return [result.expanded]

    elif strategy == "decompose":
        result = decompose_query(query)
        return result.sub_queries

    elif strategy == "keywords":
        result = extract_keywords(query)
        # Use the boolean query as primary, individual keywords as fallback
        return [result.boolean_query] + result.keywords[:3]

    return [query]  # No-op fallback


CORRECTION_STRATEGIES = ["expand", "decompose", "keywords"]

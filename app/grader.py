"""
grader.py
=========
LLM-as-judge: evaluates whether retrieved documents are relevant to a query.
Also verifies that generated answers are grounded in provided documents.

Design decisions:
- Binary output (relevant/not) keeps decision logic simple
- Structured output via Pydantic ensures reliable parsing
- Reason field gives observability into why docs were rejected
- Runs per-document, not per-batch, so we get granular signals
- v1.5: Track costs via response.usage
- v1.6: Answer-level verification (grounding check after generation)
"""

import logging
import os
from typing import Optional, Tuple
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from costs import CostBreakdown
from retry import retry_grader
from errors import GraderError

logger = logging.getLogger(__name__)

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

GRADER_MODEL = "gpt-4o-mini-2024-07-18"  # Cheaper — grading is simpler than generation


class GradeResult(BaseModel):
    relevant: bool          # Is this document relevant to the query?
    score: float            # Confidence 0.0–1.0
    reason: str             # Why relevant or not (for observability)


GRADER_SYSTEM = """You are a relevance grader evaluating if a retrieved document answers a user's query.

Be strict: only return relevant=true if the document DIRECTLY contains information that would help answer the query.
A document about a related topic but missing key details should be relevant=false.

Return:
- relevant: true/false
- score: 0.0 (completely irrelevant) to 1.0 (perfectly relevant)
- reason: one sentence explaining your decision"""


@retry_grader(max_retries=3)
def grade_document(query: str, document: str) -> Tuple[GradeResult, Optional[CostBreakdown]]:
    """
    Grade a single document's relevance to a query.

    Returns:
        (grade_result: GradeResult, cost_breakdown: CostBreakdown)

    Retry behavior:
        - Retries on transient errors (rate limits, 5xx, timeouts)
        - Fails fast on validation errors (4xx)
        - Max 3 attempts with exponential backoff
    """
    prompt = f"""Query: {query}

Document:
{document[:2000]}

Is this document relevant to answering the query?"""

    try:
        response = client.beta.chat.completions.parse(
            model=GRADER_MODEL,
            messages=[
                {"role": "system", "content": GRADER_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            response_format=GradeResult,
        )

        grade = response.choices[0].message.parsed

        # v1.5: Track cost
        cost = CostBreakdown(
            model=GRADER_MODEL,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            cost_usd=0,  # will be calculated in __post_init__
        )

        return grade, cost

    except ValueError as e:
        logger.error(f"Failed to parse grader response: {str(e)}")
        # Fallback to neutral grade on parse error
        return (
            GradeResult(relevant=False, score=0.5, reason="Grader parse error"),
            None,
        )


def grade_documents(query: str, documents: list[str]):
    """
    Grade multiple documents, return (doc, grade, cost) tuples.

    v1.5: Now returns costs for cost tracking.
    """
    results = []
    for doc in documents:
        grade, cost = grade_document(query, doc)
        results.append((doc, grade, cost))
    return results


# ---------------------------------------------------------------------------
# v1.6: Answer-level verification — checks if answer is grounded in docs
# ---------------------------------------------------------------------------

class AnswerVerification(BaseModel):
    grounded: bool          # Is the answer supported by the provided documents?
    confidence: float       # 0.0–1.0: how well-supported is the answer?
    gaps: list[str]         # Claims in the answer NOT found in documents
    supported_claims: int   # How many answer claims are document-backed


VERIFIER_SYSTEM = """You are a fact-checking assistant verifying if an AI-generated answer
is grounded in the provided source documents.

Check each factual claim in the answer against the documents.

Return:
- grounded: true if the answer's main claims are supported by documents, false if it contains unsupported claims
- confidence: 0.0 (completely unsupported) to 1.0 (fully supported by documents)
- gaps: list of specific claims in the answer that are NOT found in or contradict the documents (empty list if none)
- supported_claims: count of claims you could verify against the documents"""


@retry_grader(max_retries=3)
def verify_answer(
    query: str,
    answer: str,
    documents: list[str],
) -> tuple["AnswerVerification", Optional[CostBreakdown]]:
    """
    v1.6: Verify the generated answer is grounded in the provided documents.

    This is the second quality gate — CRAG grades documents before generation,
    but this checks the answer itself after generation.

    Returns:
        (verification: AnswerVerification, cost: CostBreakdown)

    Retry behavior:
        - Retries on transient errors (rate limits, 5xx, timeouts)
        - Returns unverified fallback on persistent failures
        - Max 3 attempts with exponential backoff
    """
    if not documents or not answer:
        return AnswerVerification(
            grounded=False,
            confidence=0.0,
            gaps=["No documents available to verify against"],
            supported_claims=0,
        ), None

    doc_block = "\n\n".join(
        f"[Doc {i+1}]:\n{doc[:800]}" for i, doc in enumerate(documents)
    )

    prompt = f"""Query: {query}

Source Documents:
{doc_block}

Generated Answer:
{answer}

Verify if the answer is grounded in the source documents above."""

    try:
        response = client.beta.chat.completions.parse(
            model=GRADER_MODEL,
            messages=[
                {"role": "system", "content": VERIFIER_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            response_format=AnswerVerification,
        )

        verification = response.choices[0].message.parsed
        cost = CostBreakdown(
            model=GRADER_MODEL,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            cost_usd=0,
        )

        return verification, cost

    except ValueError as e:
        logger.error(f"Failed to parse verifier response: {str(e)}")
        # Fallback: return unverified but don't block answer
        return (
            AnswerVerification(
                grounded=False,
                confidence=0.0,
                gaps=["Verification failed"],
                supported_claims=0,
            ),
            None,
        )


def filter_relevant(
    query: str,
    documents: list[str],
    threshold: float = 0.5,
) -> tuple[list[str], list[GradeResult], list[CostBreakdown]]:
    """
    Return only relevant documents, all grade results, and cost breakdowns.

    Args:
        query: The user's question
        documents: Retrieved documents to grade
        threshold: Minimum score to keep a document (default 0.5)

    Returns:
        (relevant_docs, all_grades, all_costs)  [v1.5: added costs]
    """
    graded = grade_documents(query, documents)
    relevant_docs = [
        doc for doc, grade, cost in graded
        if grade.relevant and grade.score >= threshold
    ]
    all_grades = [grade for _, grade, cost in graded]
    all_costs = [cost for _, grade, cost in graded]
    return relevant_docs, all_grades, all_costs

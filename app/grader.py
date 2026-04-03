"""
grader.py
=========
LLM-as-judge: evaluates whether retrieved documents are relevant to a query.

Design decisions:
- Binary output (relevant/not) keeps decision logic simple
- Structured output via Pydantic ensures reliable parsing
- Reason field gives observability into why docs were rejected
- Runs per-document, not per-batch, so we get granular signals
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

GRADER_MODEL = "gpt-4o-mini"  # Cheaper — grading is simpler than generation


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


def grade_document(query: str, document: str) -> GradeResult:
    """Grade a single document's relevance to a query."""
    prompt = f"""Query: {query}

Document:
{document[:2000]}

Is this document relevant to answering the query?"""

    response = client.beta.chat.completions.parse(
        model=GRADER_MODEL,
        messages=[
            {"role": "system", "content": GRADER_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        response_format=GradeResult,
    )
    return response.choices[0].message.parsed


def grade_documents(query: str, documents: list[str]) -> list[tuple[str, GradeResult]]:
    """Grade multiple documents, return (doc, grade) pairs."""
    return [(doc, grade_document(query, doc)) for doc in documents]


def filter_relevant(
    query: str,
    documents: list[str],
    threshold: float = 0.5,
) -> tuple[list[str], list[GradeResult]]:
    """
    Return only relevant documents and all grade results.

    Args:
        query: The user's question
        documents: Retrieved documents to grade
        threshold: Minimum score to keep a document (default 0.5)

    Returns:
        (relevant_docs, all_grades)
    """
    graded = grade_documents(query, documents)
    relevant_docs = [
        doc for doc, grade in graded
        if grade.relevant and grade.score >= threshold
    ]
    all_grades = [grade for _, grade in graded]
    return relevant_docs, all_grades

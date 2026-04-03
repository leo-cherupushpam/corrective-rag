"""
crag.py
=======
Core Corrective RAG pipeline.

Compares two modes:
  - BaselineRAG: retrieve → generate (no quality gate)
  - CRAG:        retrieve → grade → correct (if needed) → generate

Both use the same retriever and generator for a fair comparison.

Design decisions:
- FAISS for local vector store (no external services needed)
- OpenAI embeddings (text-embedding-3-small, fast + cheap)
- GPT-4o for generation (quality), gpt-4o-mini for grading (cost)
- Max 2 correction attempts before fallback
- Full trace logged for every query (observability)
"""

import dataclasses
import os
from dataclasses import dataclass, field
from typing import Optional

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from corrector import CORRECTION_STRATEGIES, get_correction_candidates
from grader import filter_relevant

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBED_MODEL = "text-embedding-3-small"
GENERATE_MODEL = "gpt-4o"
TOP_K = 5
MAX_CORRECTIONS = 2

GENERATOR_SYSTEM = """You are a precise question-answering assistant.
Answer the user's question using ONLY the provided documents.
If the documents don't contain enough information, say exactly: "I don't have enough information to answer this."
Always cite the source document number (e.g., [Doc 1]) when using information from it.
Keep answers concise and factual."""


# ---------------------------------------------------------------------------
# Trace — full audit log of a single query
# ---------------------------------------------------------------------------

@dataclass
class GradeTrace:
    document_preview: str
    relevant: bool
    score: float
    reason: str


@dataclass
class CorrectionTrace:
    strategy: str
    query_used: str
    docs_retrieved: int
    docs_passed_grade: int


@dataclass
class QueryTrace:
    query: str
    mode: str                               # "baseline" | "crag"
    answer: str
    docs_used: list[str] = field(default_factory=list)
    grades: list[GradeTrace] = field(default_factory=list)
    corrections: list[CorrectionTrace] = field(default_factory=list)
    needed_correction: bool = False
    fallback_used: bool = False             # True if answered without retrieved docs
    total_llm_calls: int = 0


# ---------------------------------------------------------------------------
# VectorStore — FAISS-backed retriever
# ---------------------------------------------------------------------------

class VectorStore:
    """Simple FAISS-backed vector store for document retrieval."""

    def __init__(self):
        self.index: Optional[faiss.IndexFlatL2] = None
        self.documents: list[str] = []
        self.dim = 1536  # text-embedding-3-small dimension

    def _embed(self, texts: list[str]) -> np.ndarray:
        response = client.embeddings.create(model=EMBED_MODEL, input=texts)
        vectors = [r.embedding for r in response.data]
        return np.array(vectors, dtype="float32")

    def add_documents(self, documents: list[str]):
        """Index documents into FAISS."""
        embeddings = self._embed(documents)
        if self.index is None:
            self.index = faiss.IndexFlatL2(self.dim)
        self.index.add(embeddings)
        self.documents.extend(documents)

    def retrieve(self, query: str, k: int = TOP_K) -> list[str]:
        """Retrieve top-k most similar documents for query."""
        if not self.documents:
            return []
        query_vec = self._embed([query])
        k = min(k, len(self.documents))
        _, indices = self.index.search(query_vec, k)
        return [self.documents[i] for i in indices[0] if i < len(self.documents)]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate_answer(query: str, documents: list[str]) -> str:
    """Generate an answer grounded in provided documents."""
    if not documents:
        return "I don't have enough information to answer this."

    doc_block = "\n\n".join(
        f"[Doc {i+1}]:\n{doc}" for i, doc in enumerate(documents)
    )
    response = client.chat.completions.create(
        model=GENERATE_MODEL,
        messages=[
            {"role": "system", "content": GENERATOR_SYSTEM},
            {"role": "user", "content": f"Documents:\n{doc_block}\n\nQuestion: {query}"},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Baseline RAG
# ---------------------------------------------------------------------------

def baseline_rag(query: str, store: VectorStore) -> QueryTrace:
    """Standard RAG: retrieve → generate (no quality gate)."""
    trace = QueryTrace(query=query, mode="baseline", answer="")

    docs = store.retrieve(query)
    trace.docs_used = docs
    trace.total_llm_calls = 1  # just generation

    trace.answer = generate_answer(query, docs)
    return trace


# ---------------------------------------------------------------------------
# CRAG
# ---------------------------------------------------------------------------

def crag(query: str, store: VectorStore) -> QueryTrace:
    """
    Corrective RAG:
      1. Retrieve
      2. Grade — are docs relevant?
      3. If not → Correct (reformulate query, re-retrieve)
      4. Repeat up to MAX_CORRECTIONS times
      5. Generate from best docs found, or fallback
    """
    trace = QueryTrace(query=query, mode="crag", answer="")
    current_query = query
    llm_calls = 0

    for attempt in range(MAX_CORRECTIONS + 1):
        # Step 1: Retrieve
        docs = store.retrieve(current_query)

        # Step 2: Grade
        relevant_docs, grades = filter_relevant(current_query, docs)
        llm_calls += len(docs)  # one grader call per doc

        # Log grades to trace
        for doc, grade in zip(docs, grades):
            trace.grades.append(GradeTrace(
                document_preview=doc[:100] + "…",
                relevant=grade.relevant,
                score=grade.score,
                reason=grade.reason,
            ))

        if relevant_docs:
            # Docs passed grade — use them
            trace.docs_used = relevant_docs
            break

        # Step 3: Correct — try next strategy
        trace.needed_correction = True
        if attempt < MAX_CORRECTIONS:
            strategy = CORRECTION_STRATEGIES[min(attempt, len(CORRECTION_STRATEGIES) - 1)]
            candidates = get_correction_candidates(current_query, strategy)
            current_query = candidates[0]
            llm_calls += 1  # corrector call

            trace.corrections.append(CorrectionTrace(
                strategy=strategy,
                query_used=current_query,
                docs_retrieved=len(docs),
                docs_passed_grade=len(relevant_docs),
            ))
        else:
            # All corrections exhausted — fallback
            trace.fallback_used = True
            trace.docs_used = []

    # Step 4: Generate
    trace.answer = generate_answer(query, trace.docs_used)
    trace.total_llm_calls = llm_calls + 1  # +1 for generation

    return trace


# ---------------------------------------------------------------------------
# Convenience: run both and compare
# ---------------------------------------------------------------------------

@dataclass
class ComparisonResult:
    query: str
    baseline: QueryTrace
    crag: QueryTrace

    @property
    def crag_needed_correction(self) -> bool:
        return self.crag.needed_correction

    @property
    def crag_used_fallback(self) -> bool:
        return self.crag.fallback_used

    @property
    def extra_llm_calls(self) -> int:
        return self.crag.total_llm_calls - self.baseline.total_llm_calls


def compare(query: str, store: VectorStore) -> ComparisonResult:
    """Run both Baseline RAG and CRAG on the same query."""
    return ComparisonResult(
        query=query,
        baseline=baseline_rag(query, store),
        crag=crag(query, store),
    )

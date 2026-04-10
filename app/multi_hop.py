"""
multi_hop.py
============
Multi-hop retrieval: detect when a query needs multiple documents and retrieve them sequentially.

Design:
  - After single-hop retrieval, check: are these docs sufficient?
  - If not: extract "bridge entity" (what's missing), issue sub-query
  - Rerank + grade the bridge docs, merge with initial docs
  - Up to 2 hops max to control cost

v2.0: Multi-hop Retrieval
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from costs import CostBreakdown, calculate_cost
from grader import filter_relevant
from reranker import rerank_documents, should_rerank
from retry import retry_corrector
from errors import CorrectionError

logger = logging.getLogger(__name__)

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DETECTOR_MODEL = "gpt-4o-mini-2024-07-18"  # Consistent with grader/corrector
MAX_HOPS = 2
TOP_K = 10  # Reuse retrieval top-k from crag.py


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

class MultiHopDecision(BaseModel):
    """LLM-as-judge: decide if multi-hop is needed and extract bridge query."""
    needs_multi_hop: bool
    bridge_query: str      # Sub-query to retrieve missing info ("" if not needed)
    bridge_entity: str     # Human-readable description of what's missing
    reason: str            # Why multi-hop is/isn't needed


@dataclass
class MultiHopTrace:
    """Trace of a single hop in multi-hop retrieval."""
    hop_number: int                         # 1, 2, ...
    bridge_query: str                       # Query used for this hop
    bridge_entity: str                      # What we were looking for
    docs_retrieved: int                     # Raw docs returned from retriever
    docs_passed_grade: int                  # Docs that passed relevance grade
    docs_added: list[str] = field(default_factory=list)  # Actual doc texts merged in
    cost_breakdown: list[CostBreakdown] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Multi-hop Detector
# ---------------------------------------------------------------------------

DETECTOR_SYSTEM = """You are a question decomposition specialist.
Given a user's question and the documents retrieved so far, decide:
1. Are these documents sufficient to fully answer the question?
2. If not, what bridging concept or information is missing?
3. What sub-query would retrieve that missing information?

Be strict: only recommend multi-hop if the initial docs genuinely lack a key concept.
For example:
- Q: "Can I return an item if it arrived after 30 days?"
  Docs: [Return Policy (says 30 days)] + [Shipping (says 5-7 days)]
  → Decision: needs_multi_hop = True, bridge = "delivery delays/exceptions to return window"
- Q: "What is your return policy?"
  Docs: [Return Policy doc]
  → Decision: needs_multi_hop = False (already have the answer)"""


@retry_corrector(max_retries=2)
def detect_multi_hop(query: str, docs: list[str]) -> tuple[MultiHopDecision, Optional[CostBreakdown]]:
    """
    LLM call: decide if query needs multi-hop retrieval + extract bridge query.

    Args:
        query: User's question
        docs: Documents retrieved so far

    Returns:
        (decision: MultiHopDecision, cost: CostBreakdown)

    Retry behavior:
        - Retries on rate limits, 5xx errors, timeouts
        - Gracefully returns no-multi-hop on persistent failure
        - Max 2 attempts with exponential backoff
    """
    doc_preview = "\n---\n".join([d[:300] + "..." if len(d) > 300 else d for d in docs])

    prompt = f"""Question: {query}

Retrieved documents so far:
{doc_preview}

Decide: Do these documents fully answer the question, or is a bridging concept missing?
If multi-hop is needed, what should the follow-up query be?"""

    try:
        response = client.beta.chat.completions.parse(
            model=DETECTOR_MODEL,
            messages=[
                {"role": "system", "content": DETECTOR_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            response_format=MultiHopDecision,
            temperature=0,
        )

        decision = response.choices[0].message.parsed
        cost = CostBreakdown(
            model=DETECTOR_MODEL,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            cost_usd=0,  # will be calculated in __post_init__
        )
        return decision, cost

    except ValueError as e:
        logger.error(f"Multi-hop detection parse error: {str(e)}")
        # Fallback: no multi-hop on parse error
        return MultiHopDecision(
            needs_multi_hop=False,
            bridge_query="",
            bridge_entity="",
            reason=f"Detection parse error"
        ), None

    except Exception as e:
        logger.error(f"Multi-hop detection error: {str(e)}")
        # Fallback: no multi-hop on error
        return MultiHopDecision(
            needs_multi_hop=False,
            bridge_query="",
            bridge_entity="",
            reason=f"Detection error"
        ), None


# ---------------------------------------------------------------------------
# Document Deduplication
# ---------------------------------------------------------------------------

def deduplicate_docs(docs1: list[str], docs2: list[str], threshold: float = 0.8) -> list[str]:
    """
    Remove near-duplicates: if doc2[i] is >80% similar to any doc in docs1, skip it.
    Uses simple character overlap as heuristic (not semantic).
    """
    if not docs2:
        return docs1

    result = docs1.copy()
    for doc2 in docs2:
        # Skip if doc2 is almost identical to something in result (80% overlap)
        is_duplicate = False
        for doc1 in result:
            overlap = len(set(doc2.split()) & set(doc1.split())) / max(len(set(doc2.split())), 1)
            if overlap > threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            result.append(doc2)

    return result


# ---------------------------------------------------------------------------
# Multi-hop Orchestrator
# ---------------------------------------------------------------------------

def multi_hop_retrieve(
    query: str,
    initial_docs: list[str],
    store,  # VectorStore
    max_hops: int = MAX_HOPS,
) -> tuple[list[str], list[MultiHopTrace], list[CostBreakdown]]:
    """
    Orchestrates multi-hop retrieval:
    1. detect_multi_hop(query, current_docs)
    2. If needed: retrieve(bridge_query) → rerank → grade → merge
    3. Repeat up to max_hops times

    Returns:
        (merged_docs: list[str], traces: list[MultiHopTrace], all_costs: list[CostBreakdown])
    """
    all_docs = initial_docs.copy()
    all_traces = []
    all_costs = []

    current_query = query
    for hop_num in range(1, max_hops + 1):
        # Detect if multi-hop is needed
        decision, detection_cost = detect_multi_hop(current_query, all_docs)
        if detection_cost:
            all_costs.append(detection_cost)

        if not decision.needs_multi_hop:
            # No further hops needed
            break

        # Retrieve bridge docs
        bridge_docs = store.retrieve(decision.bridge_query, top_k=TOP_K)

        # Rerank if we got many docs
        if should_rerank(bridge_docs):
            bridge_docs = rerank_documents(decision.bridge_query, bridge_docs, k=3)

        # Grade the bridge docs
        relevant_bridge_docs, grades, grade_costs = filter_relevant(decision.bridge_query, bridge_docs)
        all_costs.extend(grade_costs)

        # Deduplicate: don't add docs we already have
        merged = deduplicate_docs(all_docs, relevant_bridge_docs)
        newly_added = [d for d in relevant_bridge_docs if d in merged and d not in all_docs]

        # Record trace
        trace = MultiHopTrace(
            hop_number=hop_num,
            bridge_query=decision.bridge_query,
            bridge_entity=decision.bridge_entity,
            docs_retrieved=len(bridge_docs),
            docs_passed_grade=len(relevant_bridge_docs),
            docs_added=newly_added,
            cost_breakdown=grade_costs,
        )
        all_traces.append(trace)
        all_docs = merged

        # Prepare for next iteration (in case loop continues)
        current_query = decision.bridge_query

    return all_docs, all_traces, all_costs

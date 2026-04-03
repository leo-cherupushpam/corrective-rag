"""
eval.py
=======
Evaluates Baseline RAG vs. CRAG on a test set.

Purpose:
  The whole point of CRAG is to reduce hallucinations.
  This script makes that measurable, not just claimed.

Evaluation approach:
  - Load test documents (knowledge base)
  - Load test Q&A pairs (questions + expected answers)
  - Run both Baseline RAG and CRAG on each question
  - Score: does the answer contain expected key facts?
  - Report: hallucination rate, correction rate, cost delta

Run:
  python eval.py

Output:
  Baseline vs. CRAG comparison table + per-question breakdown
"""

import json
import os
import time

from crag import VectorStore, baseline_rag, crag
from costs import format_cost  # v1.5: cost formatting

# ---------------------------------------------------------------------------
# Test knowledge base: company FAQ documents
# ---------------------------------------------------------------------------

DOCUMENTS = [
    """Return Policy:
    We accept returns within 30 days of purchase. Items must be unused and in original packaging.
    To initiate a return, contact support@example.com with your order number.
    Refunds are processed within 5–7 business days after we receive the item.
    We do not accept returns on digital products or sale items.""",

    """Shipping Information:
    Standard shipping takes 5–7 business days. Express shipping takes 2–3 business days.
    Free shipping is available on orders over $50. We ship to all 50 US states.
    International shipping is not currently available. Tracking numbers are emailed upon shipment.""",

    """Subscription Plans:
    We offer three plans: Basic ($9/mo), Pro ($29/mo), and Enterprise ($99/mo).
    Basic includes 5 projects, 10GB storage. Pro includes unlimited projects, 100GB storage.
    Enterprise adds priority support, SSO, and custom integrations.
    All plans include a 14-day free trial. Cancel anytime — no penalty.""",

    """Privacy Policy Summary:
    We collect email, name, and usage data to provide our service.
    We never sell personal data to third parties.
    Users can request data deletion by emailing privacy@example.com.
    We use industry-standard encryption for all data at rest and in transit.
    Cookie data is used for analytics only and is anonymized after 90 days.""",

    """Support and Contact:
    Support is available Monday–Friday, 9am–6pm EST.
    Email: support@example.com. Average response time: 4 hours.
    Live chat is available on Pro and Enterprise plans.
    For billing questions, contact billing@example.com.
    Our knowledge base at docs.example.com covers most common questions.""",

    """Cancellation Policy:
    You can cancel your subscription at any time from Account Settings.
    Cancellation takes effect at the end of the current billing period.
    We do not offer prorated refunds for mid-cycle cancellations.
    Your data is retained for 30 days after cancellation before deletion.""",
]

# ---------------------------------------------------------------------------
# Test Q&A pairs
# Format: (question, expected_key_facts, is_answerable)
# is_answerable=False tests that CRAG correctly says "I don't know"
# ---------------------------------------------------------------------------

TEST_CASES = [
    {
        "question": "What is your return policy?",
        "expected_facts": ["30 days", "unused", "5-7 business days"],
        "answerable": True,
    },
    {
        "question": "How long does standard shipping take?",
        "expected_facts": ["5-7 business days", "5–7 business days"],
        "answerable": True,
    },
    {
        "question": "What does the Pro plan cost?",
        "expected_facts": ["$29", "29/mo"],
        "answerable": True,
    },
    {
        "question": "Do you ship internationally?",
        "expected_facts": ["not currently", "not available", "international"],
        "answerable": True,
    },
    {
        "question": "What are your support hours?",
        "expected_facts": ["Monday", "Friday", "9am", "6pm"],
        "answerable": True,
    },
    {
        "question": "Can I get a refund if I cancel mid-month?",
        "expected_facts": ["prorated", "no", "billing period"],
        "answerable": True,
    },
    {
        "question": "Do you support OAuth 2.0 authentication?",  # Not in docs
        "expected_facts": [],
        "answerable": False,
    },
    {
        "question": "What is your revenue this quarter?",  # Not in docs
        "expected_facts": [],
        "answerable": False,
    },
]


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

def score_answer(answer: str, expected_facts: list[str], answerable: bool) -> dict:
    """
    Simple keyword-based scorer.
    For a PM portfolio: shows the evaluation methodology.
    For production: replace with LLM-as-judge or human eval.
    """
    answer_lower = answer.lower()

    if not answerable:
        # Correct behavior: say "don't know" or similar
        fallback_phrases = [
            "don't have enough information",
            "i don't know",
            "not in the provided documents",
            "cannot find",
            "no information",
            "i don't have",
        ]
        correctly_abstained = any(p in answer_lower for p in fallback_phrases)
        return {
            "answerable": False,
            "correctly_abstained": correctly_abstained,
            "hallucinated": not correctly_abstained,
            "facts_found": 0,
            "facts_expected": 0,
        }

    # Answerable: check if key facts are present
    facts_found = sum(1 for fact in expected_facts if fact.lower() in answer_lower)
    return {
        "answerable": True,
        "correctly_abstained": False,
        "hallucinated": facts_found == 0 and len(expected_facts) > 0,
        "facts_found": facts_found,
        "facts_expected": len(expected_facts),
    }


# ---------------------------------------------------------------------------
# Main evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation():
    print("=" * 60)
    print("CORRECTIVE RAG EVALUATION")
    print("Baseline RAG  vs.  CRAG")
    print("=" * 60)
    print()

    # Setup vector store
    print("Indexing documents...")
    store = VectorStore()
    store.add_documents(DOCUMENTS)
    print(f"✅ {len(DOCUMENTS)} documents indexed\n")

    results = []
    baseline_hallucinations = 0
    crag_hallucinations = 0
    crag_corrections = 0
    crag_fallbacks = 0
    total_extra_calls = 0
    total_baseline_cost = 0  # v1.5: cost tracking
    total_crag_cost = 0      # v1.5: cost tracking

    for i, tc in enumerate(TEST_CASES):
        q = tc["question"]
        print(f"Q{i+1}: {q}")

        # Run baseline
        b_trace = baseline_rag(q, store)
        b_score = score_answer(b_trace.answer, tc["expected_facts"], tc["answerable"])

        # Small delay to avoid rate limits
        time.sleep(0.5)

        # Run CRAG
        c_trace = crag(q, store)
        c_score = score_answer(c_trace.answer, tc["expected_facts"], tc["answerable"])

        # Track metrics
        if b_score["hallucinated"]:
            baseline_hallucinations += 1
        if c_score["hallucinated"]:
            crag_hallucinations += 1
        if c_trace.needed_correction:
            crag_corrections += 1
        if c_trace.fallback_used:
            crag_fallbacks += 1
        total_extra_calls += c_trace.total_llm_calls - b_trace.total_llm_calls

        # v1.5: Track costs
        total_baseline_cost += b_trace.total_cost_usd
        total_crag_cost += c_trace.total_cost_usd

        results.append({
            "question": q,
            "answerable": tc["answerable"],
            "baseline_answer": b_trace.answer[:120] + "…",
            "crag_answer": c_trace.answer[:120] + "…",
            "baseline_hallucinated": b_score["hallucinated"],
            "crag_hallucinated": c_score["hallucinated"],
            "crag_needed_correction": c_trace.needed_correction,
            "crag_fallback": c_trace.fallback_used,
            "extra_llm_calls": c_trace.total_llm_calls - b_trace.total_llm_calls,
            "baseline_cost_usd": round(b_trace.total_cost_usd, 6),  # v1.5
            "crag_cost_usd": round(c_trace.total_cost_usd, 6),      # v1.5
        })

        status_b = "❌ Hallucinated" if b_score["hallucinated"] else "✅ Correct"
        status_c = "❌ Hallucinated" if c_score["hallucinated"] else "✅ Correct"
        correction_note = " (corrected)" if c_trace.needed_correction else ""
        print(f"  Baseline: {status_b}")
        print(f"  CRAG:     {status_c}{correction_note}")
        print()

    # Summary
    total = len(TEST_CASES)
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<35} {'Baseline':>10} {'CRAG':>10}")
    print("-" * 60)
    print(f"{'Hallucinations':<35} {baseline_hallucinations:>10} {crag_hallucinations:>10}")
    print(f"{'Hallucination Rate':<35} {baseline_hallucinations/total*100:>9.1f}% {crag_hallucinations/total*100:>9.1f}%")
    print(f"{'Queries needing correction':<35} {'N/A':>10} {crag_corrections:>10}")
    print(f"{'Correction fallback used':<35} {'N/A':>10} {crag_fallbacks:>10}")
    print(f"{'Avg extra LLM calls (CRAG)':<35} {'0':>10} {total_extra_calls/total:>9.1f}")
    # v1.5: Add cost metrics
    print(f"{'Total cost':<35} {format_cost(total_baseline_cost):>10} {format_cost(total_crag_cost):>10}")
    print(f"{'Avg cost per query':<35} {format_cost(total_baseline_cost/total):>10} {format_cost(total_crag_cost/total):>10}")
    print()

    reduction = (baseline_hallucinations - crag_hallucinations) / max(baseline_hallucinations, 1) * 100
    print(f"Hallucination reduction: {reduction:.0f}%")
    cost_delta = total_crag_cost - total_baseline_cost
    print(f"Cost delta: {format_cost(cost_delta)} ({cost_delta/total_baseline_cost*100:+.0f}%)")
    print()

    # Save results to JSON for dashboard
    # v1.5: Include cost metrics
    with open("eval_results.json", "w") as f:
        json.dump({
            "summary": {
                "total_questions": total,
                "baseline_hallucinations": baseline_hallucinations,
                "crag_hallucinations": crag_hallucinations,
                "baseline_hallucination_rate": round(baseline_hallucinations / total * 100, 1),
                "crag_hallucination_rate": round(crag_hallucinations / total * 100, 1),
                "hallucination_reduction_pct": round(reduction, 1),
                "crag_corrections": crag_corrections,
                "crag_fallbacks": crag_fallbacks,
                "avg_extra_llm_calls": round(total_extra_calls / total, 1),
                # v1.5: Cost metrics
                "total_baseline_cost_usd": round(total_baseline_cost, 6),
                "total_crag_cost_usd": round(total_crag_cost, 6),
                "avg_baseline_cost_per_query": round(total_baseline_cost / total, 6),
                "avg_crag_cost_per_query": round(total_crag_cost / total, 6),
                "cost_delta_usd": round(total_crag_cost - total_baseline_cost, 6),
                "cost_delta_pct": round((total_crag_cost - total_baseline_cost) / total_baseline_cost * 100, 1),
            },
            "per_question": results,
        }, f, indent=2)

    print("Results saved to eval_results.json")
    return results


if __name__ == "__main__":
    run_evaluation()

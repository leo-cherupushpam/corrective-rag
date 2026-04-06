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
  python eval.py                          # Use built-in test cases
  python eval.py --batch path/to/eval.csv # Load from CSV

CSV Format:
  question,expected_facts,answerable,category
  "What is your return policy?","30 days;unused;5-7 business days",true,direct
  "OAuth support?","",false,unanswerable

Output:
  Baseline vs. CRAG comparison table + per-question breakdown + confidence calibration
"""

import argparse
import csv
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
# Test Q&A pairs (v1.6: expanded from 8 → 20 cases)
# Format: (question, expected_key_facts, is_answerable, category)
# is_answerable=False: CRAG should say "I don't know", not hallucinate
# Categories: direct | inference | adversarial | unanswerable
# ---------------------------------------------------------------------------

TEST_CASES = [
    # --- Straightforward answerable questions ---
    {
        "question": "What is your return policy?",
        "expected_facts": ["30 days", "unused", "5-7 business days"],
        "answerable": True,
        "category": "direct",
    },
    {
        "question": "How long does standard shipping take?",
        "expected_facts": ["5-7 business days", "5–7 business days"],
        "answerable": True,
        "category": "direct",
    },
    {
        "question": "What does the Pro plan cost?",
        "expected_facts": ["$29", "29/mo"],
        "answerable": True,
        "category": "direct",
    },
    {
        "question": "Do you ship internationally?",
        "expected_facts": ["not currently", "not available", "international"],
        "answerable": True,
        "category": "direct",
    },
    {
        "question": "What are your support hours?",
        "expected_facts": ["Monday", "Friday", "9am", "6pm"],
        "answerable": True,
        "category": "direct",
    },
    {
        "question": "Can I get a refund if I cancel mid-month?",
        "expected_facts": ["prorated", "no", "billing period"],
        "answerable": True,
        "category": "direct",
    },

    # --- Require inference / locating specific details ---
    {
        "question": "What email do I use to start a return?",
        "expected_facts": ["support@example.com"],
        "answerable": True,
        "category": "inference",
    },
    {
        "question": "How long is my data kept after I cancel?",
        "expected_facts": ["30 days"],
        "answerable": True,
        "category": "inference",
    },
    {
        "question": "Which plans include live chat support?",
        "expected_facts": ["Pro", "Enterprise"],
        "answerable": True,
        "category": "inference",
    },
    {
        "question": "What storage does the Basic plan include?",
        "expected_facts": ["10GB"],
        "answerable": True,
        "category": "inference",
    },
    {
        "question": "How do I delete my personal data?",
        "expected_facts": ["privacy@example.com"],
        "answerable": True,
        "category": "inference",
    },
    {
        "question": "Is there a free trial and how long is it?",
        "expected_facts": ["14-day", "14 day"],
        "answerable": True,
        "category": "inference",
    },

    # --- Adversarial / tricky (conflated or double-negative) ---
    {
        "question": "Can I return a digital product I bought on sale?",
        "expected_facts": ["digital products", "sale items", "not"],
        "answerable": True,
        "category": "adversarial",
    },
    {
        "question": "Is express shipping free for orders over $50?",
        "expected_facts": ["free shipping", "standard", "$50"],
        "answerable": True,
        "category": "adversarial",
    },

    # --- Not in documents (should say "I don't know", not hallucinate) ---
    {
        "question": "Do you support OAuth 2.0 authentication?",
        "expected_facts": [],
        "answerable": False,
        "category": "unanswerable",
    },
    {
        "question": "What is your revenue this quarter?",
        "expected_facts": [],
        "answerable": False,
        "category": "unanswerable",
    },
    {
        "question": "Do you have a mobile app?",
        "expected_facts": [],
        "answerable": False,
        "category": "unanswerable",
    },
    {
        "question": "What is the name of your CEO?",
        "expected_facts": [],
        "answerable": False,
        "category": "unanswerable",
    },
    {
        "question": "Can I pay with cryptocurrency?",
        "expected_facts": [],
        "answerable": False,
        "category": "unanswerable",
    },
    {
        "question": "Do you offer a student discount?",
        "expected_facts": [],
        "answerable": False,
        "category": "unanswerable",
    },

    # --- Multi-hop: require bridging 2+ documents ---
    {
        "question": "Can I get live chat support if I'm on the Basic plan?",
        "expected_facts": ["Basic", "live chat", "Pro", "Enterprise"],
        "answerable": True,
        "category": "multi_hop",
    },
    {
        "question": "What happens to my projects if I cancel and my data is deleted?",
        "expected_facts": ["30 days", "cancel", "data", "Account Settings"],
        "answerable": True,
        "category": "multi_hop",
    },
    {
        "question": "Is Express shipping free for orders under $50?",
        "expected_facts": ["Express", "free shipping", "$50"],
        "answerable": True,
        "category": "multi_hop",
    },
    {
        "question": "Can I return an item if it arrived after 30 days due to slow standard shipping?",
        "expected_facts": ["30 days", "5-7 business days", "return", "shipping"],
        "answerable": True,
        "category": "multi_hop",
    },
    {
        "question": "Does Enterprise plan include priority support via email?",
        "expected_facts": ["Enterprise", "priority support", "support@example.com"],
        "answerable": True,
        "category": "multi_hop",
    },
]


# ---------------------------------------------------------------------------
# Batch CSV Loader
# ---------------------------------------------------------------------------

def load_test_cases_from_csv(csv_path: str) -> list[dict]:
    """
    Load test cases from CSV file.

    CSV format:
      question,expected_facts,answerable,category
      "What is your return policy?","30 days;unused;5-7 business days",true,direct

    expected_facts: semicolon-separated list of facts to check
    answerable: 'true' or 'false'
    category: direct|inference|adversarial|unanswerable (optional)
    """
    test_cases = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader, 1):
                if not row.get('question'):
                    print(f"⚠️  Skipping row {i}: missing question")
                    continue

                # Parse expected facts (semicolon-separated)
                facts_str = row.get('expected_facts', '')
                expected_facts = [f.strip() for f in facts_str.split(';') if f.strip()] if facts_str else []

                # Parse answerable (string 'true'/'false' → bool)
                answerable_str = row.get('answerable', 'true').lower()
                answerable = answerable_str in ('true', 'yes', '1')

                test_cases.append({
                    "question": row['question'].strip(),
                    "expected_facts": expected_facts,
                    "answerable": answerable,
                    "category": row.get('category', 'direct').strip(),
                })

        print(f"✅ Loaded {len(test_cases)} test cases from {csv_path}\n")
        return test_cases
    except FileNotFoundError:
        print(f"❌ File not found: {csv_path}")
        return []
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return []


# ---------------------------------------------------------------------------
# Confidence Calibration Analysis
# ---------------------------------------------------------------------------

def analyze_calibration(results: list[dict]) -> dict:
    """
    Analyze if confidence scores are calibrated.

    Calibration check: when CRAG predicts X% confidence, is it right X% of the time?

    Returns:
      {
        "calibration_bins": [
          {"confidence_range": "0.0-0.2", "predicted": 0.1, "actual_accuracy": 0.0, "count": 3},
          ...
        ],
        "overall_calibration_error": 0.05,
        "is_well_calibrated": true
      }
    """
    # Group results by confidence bins
    bins = {}
    bin_size = 0.1  # 10% bins

    for r in results:
        confidence = r.get('crag_answer_confidence', 0.0)
        bin_idx = int(confidence / bin_size)
        bin_key = f"{bin_idx * bin_size:.1f}-{(bin_idx + 1) * bin_size:.1f}"

        if bin_key not in bins:
            bins[bin_key] = {"confidences": [], "correct": []}

        bins[bin_key]["confidences"].append(confidence)
        # Correct if not hallucinated
        bins[bin_key]["correct"].append(not r.get('crag_hallucinated', False))

    # Calculate calibration per bin
    calibration_bins = []
    errors = []
    for bin_key in sorted(bins.keys()):
        data = bins[bin_key]
        avg_confidence = sum(data["confidences"]) / len(data["confidences"])
        accuracy = sum(data["correct"]) / len(data["correct"]) if data["correct"] else 0.0
        error = abs(avg_confidence - accuracy)
        errors.append(error)

        calibration_bins.append({
            "confidence_range": bin_key,
            "predicted_confidence": round(avg_confidence, 2),
            "actual_accuracy": round(accuracy, 2),
            "count": len(data["correct"]),
            "calibration_error": round(error, 2),
        })

    overall_error = sum(errors) / len(errors) if errors else 0.0
    is_well_calibrated = overall_error < 0.15  # <15% error is good

    return {
        "calibration_bins": calibration_bins,
        "overall_calibration_error": round(overall_error, 2),
        "is_well_calibrated": is_well_calibrated,
        "interpretation": (
            "✅ Well calibrated" if is_well_calibrated
            else "⚠️  Needs calibration" if overall_error < 0.25
            else "❌ Poorly calibrated"
        ),
    }


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

def run_evaluation(test_cases: list[dict] = None):
    """
    Run evaluation on test cases.

    Args:
        test_cases: List of dicts with keys: question, expected_facts, answerable, category.
                   If None, uses built-in TEST_CASES.
    """
    if test_cases is None:
        test_cases = TEST_CASES

    print("=" * 60)
    print("CORRECTIVE RAG EVALUATION")
    print("Baseline RAG  vs.  CRAG")
    print("=" * 60)
    print()

    # Setup vector store
    print("Indexing documents...")
    store = VectorStore()
    store.add_documents(DOCUMENTS)
    print(f"✅ {len(DOCUMENTS)} documents indexed")
    print(f"📊 Running evaluation on {len(test_cases)} test cases\n")

    results = []
    baseline_hallucinations = 0
    crag_hallucinations = 0
    crag_corrections = 0
    crag_fallbacks = 0
    crag_multi_hops = 0      # v2.0: count multi-hop queries
    total_extra_calls = 0
    total_baseline_cost = 0  # v1.5: cost tracking
    total_crag_cost = 0      # v1.5: cost tracking

    for i, tc in enumerate(test_cases):
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
        if c_trace.multi_hop_needed:  # v2.0: track multi-hop
            crag_multi_hops += 1
        total_extra_calls += c_trace.total_llm_calls - b_trace.total_llm_calls

        # v1.5: Track costs
        total_baseline_cost += b_trace.total_cost_usd
        total_crag_cost += c_trace.total_cost_usd

        results.append({
            "question": q,
            "category": tc.get("category", "direct"),    # v1.6: question category
            "answerable": tc["answerable"],
            "baseline_answer": b_trace.answer[:120] + "…",
            "crag_answer": c_trace.answer[:120] + "…",
            "baseline_hallucinated": b_score["hallucinated"],
            "crag_hallucinated": c_score["hallucinated"],
            "crag_needed_correction": c_trace.needed_correction,
            "crag_fallback": c_trace.fallback_used,
            "extra_llm_calls": c_trace.total_llm_calls - b_trace.total_llm_calls,
            "baseline_cost_usd": round(b_trace.total_cost_usd, 6),
            "crag_cost_usd": round(c_trace.total_cost_usd, 6),
            # v1.5: Confidence scores
            "baseline_answer_confidence": round(b_trace.answer_confidence, 2),
            "crag_answer_confidence": round(c_trace.answer_confidence, 2),
            "crag_grader_confidence": round(c_trace.grader_confidence, 2),
            # v1.6: Answer-level verification
            "crag_answer_grounded": c_trace.answer_grounded,
            "crag_answer_gaps": c_trace.answer_gaps,
            "crag_supported_claims": c_trace.answer_supported_claims,
            # v2.0: Multi-hop
            "crag_multi_hop_needed": c_trace.multi_hop_needed,
            "crag_multi_hop_count": len(c_trace.multi_hop_hops),
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
    print(f"{'Multi-hop queries':<35} {'N/A':>10} {crag_multi_hops:>10} (v2.0)")
    print(f"{'Avg extra LLM calls (CRAG)':<35} {'0':>10} {total_extra_calls/total:>9.1f}")
    # v1.5: Add cost metrics
    print(f"{'Total cost':<35} {format_cost(total_baseline_cost):>10} {format_cost(total_crag_cost):>10}")
    print(f"{'Avg cost per query':<35} {format_cost(total_baseline_cost/total):>10} {format_cost(total_crag_cost/total):>10}")
    # v1.5: Add confidence metrics
    avg_baseline_confidence = sum(r.get('baseline_answer_confidence', 0) for r in results) / total
    avg_crag_confidence = sum(r.get('crag_answer_confidence', 0) for r in results) / total
    avg_grader_confidence = sum(r.get('crag_grader_confidence', 0) for r in results) / total
    print(f"{'Avg answer confidence':<35} {avg_baseline_confidence:>9.2f}  {avg_crag_confidence:>9.2f}")
    print(f"{'Avg grader confidence (CRAG)':<35} {'N/A':>10} {avg_grader_confidence:>9.2f}")
    print()

    reduction = (baseline_hallucinations - crag_hallucinations) / max(baseline_hallucinations, 1) * 100
    print(f"Hallucination reduction: {reduction:.0f}%")
    cost_delta = total_crag_cost - total_baseline_cost
    print(f"Cost delta: {format_cost(cost_delta)} ({cost_delta/total_baseline_cost*100:+.0f}%)")

    # v1.6: Per-category breakdown
    categories = ["direct", "inference", "adversarial", "unanswerable"]
    print()
    print("BY CATEGORY:")
    print(f"{'Category':<15} {'Count':>6} {'CRAG Hallucinated':>18}")
    print("-" * 42)
    category_summary = {}
    for cat in categories:
        cat_results = [r for r in results if r.get("category") == cat]
        cat_hallucinated = sum(1 for r in cat_results if r["crag_hallucinated"])
        category_summary[cat] = {"count": len(cat_results), "hallucinated": cat_hallucinated}
        if cat_results:
            print(f"  {cat:<13} {len(cat_results):>6} {cat_hallucinated:>18}")
    print()

    # v1.6: Grounding summary
    grounded_count = sum(1 for r in results if r.get("crag_answer_grounded") is True)
    ungrounded_count = sum(1 for r in results if r.get("crag_answer_grounded") is False)
    not_verified = sum(1 for r in results if r.get("crag_answer_grounded") is None)
    print(f"Answer grounding (v1.6 verification):")
    print(f"  Fully grounded: {grounded_count}, Gaps found: {ungrounded_count}, Not verified (fallback): {not_verified}")
    print()

    # v1.8: Confidence calibration analysis
    calibration = analyze_calibration(results)
    print("CONFIDENCE CALIBRATION ANALYSIS:")
    print(f"  {calibration['interpretation']}")
    print(f"  Overall calibration error: {calibration['overall_calibration_error']}")
    print()
    print("  Calibration by confidence bin:")
    print(f"  {'Range':<15} {'Predicted':>12} {'Actual':>12} {'Error':>10} {'Count':>8}")
    print("  " + "-" * 60)
    for bin_data in calibration['calibration_bins']:
        print(f"  {bin_data['confidence_range']:<15} "
              f"{bin_data['predicted_confidence']:>11.2f}  "
              f"{bin_data['actual_accuracy']:>11.2f}  "
              f"{bin_data['calibration_error']:>9.2f}  "
              f"{bin_data['count']:>8}")
    print()

    # Save results to JSON for dashboard
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
                "crag_multi_hops": crag_multi_hops,
                "crag_multi_hop_rate_pct": round(crag_multi_hops / total * 100, 1),
                "avg_extra_llm_calls": round(total_extra_calls / total, 1),
                # v1.5: Cost metrics
                "total_baseline_cost_usd": round(total_baseline_cost, 6),
                "total_crag_cost_usd": round(total_crag_cost, 6),
                "avg_baseline_cost_per_query": round(total_baseline_cost / total, 6),
                "avg_crag_cost_per_query": round(total_crag_cost / total, 6),
                "cost_delta_usd": round(total_crag_cost - total_baseline_cost, 6),
                "cost_delta_pct": round((total_crag_cost - total_baseline_cost) / total_baseline_cost * 100, 1),
                # v1.5: Confidence metrics
                "avg_baseline_answer_confidence": round(avg_baseline_confidence, 2),
                "avg_crag_answer_confidence": round(avg_crag_confidence, 2),
                "avg_grader_confidence": round(avg_grader_confidence, 2),
                # v1.6: Category breakdown & grounding
                "category_breakdown": category_summary,
                "answers_grounded": grounded_count,
                "answers_with_gaps": ungrounded_count,
                # v1.8: Calibration analysis
                "calibration": calibration,
            },
            "per_question": results,
        }, f, indent=2)

    print("Results saved to eval_results.json")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate CRAG vs. Baseline RAG",
        epilog="Examples:\n"
               "  python eval.py                          # Use built-in test cases\n"
               "  python eval.py --batch path/to/eval.csv # Load from CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--batch",
        type=str,
        default=None,
        help="Path to CSV file with test cases (question, expected_facts, answerable, category)"
    )

    args = parser.parse_args()

    if args.batch:
        # Load from CSV
        test_cases = load_test_cases_from_csv(args.batch)
        if not test_cases:
            print("❌ No test cases loaded. Aborting.")
            exit(1)
    else:
        # Use built-in test cases
        test_cases = TEST_CASES

    run_evaluation(test_cases)

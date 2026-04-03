"""
cost_analysis.py
================
Comparative cost analysis across grader models.

Tests the same evaluation set with different grader models to compare:
  - Hallucination rate (accuracy)
  - Cost per query
  - Grader model efficiency

Tested models:
  - gpt-5-nano-2025-08-07 (cheapest)
  - gpt-4.1-nano-2025-04-14 (mid-cost)
  - gpt-4o-mini-2024-07-18 (current, most expensive)

Run:
  python cost_analysis.py
"""

import json
import os
import subprocess
import sys
import time
from typing import Dict

from costs import CostAnalysisReport, get_model_cost_ratio


MODELS_TO_TEST = [
    "gpt-5-nano-2025-08-07",      # 3.3x cheaper input than gpt-4o-mini
    "gpt-4.1-nano-2025-04-14",    # 1.5x cheaper input than gpt-4o-mini
    "gpt-4o-mini-2024-07-18",     # baseline (current)
]


def run_eval_with_model(model: str) -> Dict:
    """
    Run evaluation with a specific grader model.

    Steps:
    1. Update grader.py GRADER_MODEL constant
    2. Run eval.py
    3. Load and return eval_results.json
    """
    print(f"\n{'='*60}")
    print(f"Testing with grader: {model}")
    print(f"{'='*60}")

    # Update grader.py to use this model
    with open("grader.py", "r") as f:
        grader_content = f.read()

    # Replace GRADER_MODEL line
    original_model_line = None
    for line in grader_content.split("\n"):
        if line.startswith("GRADER_MODEL = "):
            original_model_line = line
            break

    if not original_model_line:
        print("❌ Could not find GRADER_MODEL in grader.py")
        return None

    new_grader_content = grader_content.replace(
        original_model_line,
        f'GRADER_MODEL = "{model}"  # Cost analysis test'
    )

    with open("grader.py", "w") as f:
        f.write(new_grader_content)

    print(f"✅ Updated GRADER_MODEL to {model}")

    # Run eval.py
    print("Running evaluation...")
    result = subprocess.run(
        ["python", "eval.py"],
        capture_output=True,
        text=True,
        timeout=600
    )

    if result.returncode != 0:
        print(f"❌ Evaluation failed with return code {result.returncode}")
        print("STDERR:", result.stderr[-500:])
        return None

    # Load eval_results.json
    try:
        with open("eval_results.json", "r") as f:
            eval_results = json.load(f)
        print("✅ Evaluation complete")
        return eval_results
    except Exception as e:
        print(f"❌ Failed to load eval_results.json: {e}")
        return None
    finally:
        # Restore original grader.py
        with open("grader.py", "w") as f:
            f.write(grader_content)
        print(f"✅ Restored original grader.py")


def analyze_results(results_by_model: Dict[str, Dict]) -> Dict:
    """
    Compare evaluation results across models.

    Returns summary with:
    - Cost comparison (absolute and relative)
    - Accuracy comparison (hallucination rate, correction rate)
    - Recommendations
    """
    report = CostAnalysisReport(models_tested=list(results_by_model.keys()))

    for model, eval_result in results_by_model.items():
        summary = eval_result["summary"]

        # Calculate per-model metrics
        result_dict = {
            "hallucination_rate": summary["crag_hallucination_rate"],
            "avg_cost_per_query": summary["avg_crag_cost_per_query"],
            "total_cost": summary["total_crag_cost_usd"],
            "corrections_needed": summary["crag_corrections"],
            "fallbacks_used": summary["crag_fallbacks"],
        }
        report.add_result(model, result_dict)

    return {
        "cost_winner": report.get_cost_winner(),
        "accuracy_winner": report.get_accuracy_winner(),
        "best_tradeoff": report.get_best_tradeoff()[0],
        "report_summary": report.summary(),
        "detailed_results": results_by_model,
    }


def main():
    print("\n" + "="*60)
    print("CRAG v1.5: GRADER MODEL COST ANALYSIS")
    print("="*60)

    results_by_model = {}

    # Test each model
    for model in MODELS_TO_TEST:
        eval_result = run_eval_with_model(model)
        if eval_result:
            results_by_model[model] = eval_result
            # Wait between tests to avoid rate limits
            if model != MODELS_TO_TEST[-1]:
                print(f"\nWaiting 5s before next test...")
                time.sleep(5)

    if not results_by_model:
        print("❌ No evaluation results collected")
        sys.exit(1)

    # Analyze and report
    analysis = analyze_results(results_by_model)

    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)
    print(analysis["report_summary"])

    # Detailed comparison
    print("\n" + "="*60)
    print("DETAILED COMPARISON TABLE")
    print("="*60)

    print(f"\n{'Model':<35} {'Hallucination':>15} {'Cost/Query':>15} {'Total Cost':>15}")
    print("-" * 85)

    for model in MODELS_TO_TEST:
        if model in results_by_model:
            summary = results_by_model[model]["summary"]
            cost_per_query = summary["avg_crag_cost_per_query"]
            total_cost = summary["total_crag_cost_usd"]
            hallucination = summary["crag_hallucination_rate"]

            from costs import format_cost
            cost_str = format_cost(cost_per_query)
            total_str = format_cost(total_cost)

            print(
                f"{model:<35} {hallucination:>14.1f}% {cost_str:>15} {total_str:>15}"
            )

    # Cost savings analysis
    print("\n" + "="*60)
    print("COST SAVINGS ANALYSIS")
    print("="*60)

    baseline_model = "gpt-4o-mini-2024-07-18"
    if baseline_model in results_by_model:
        baseline_cost = results_by_model[baseline_model]["summary"]["avg_crag_cost_per_query"]

        for model in MODELS_TO_TEST:
            if model != baseline_model and model in results_by_model:
                model_cost = results_by_model[model]["summary"]["avg_crag_cost_per_query"]
                savings = baseline_cost - model_cost
                savings_pct = (savings / baseline_cost) * 100

                ratio = get_model_cost_ratio(model, baseline_model)
                ratio_pct = (1 - ratio) * 100

                from costs import format_cost
                print(f"\n{model} vs {baseline_model}:")
                print(f"  Cost delta: {format_cost(savings)} ({savings_pct:+.1f}%)")
                print(f"  Cost ratio: {ratio:.2f}x (or {ratio_pct:.0f}% savings)")

    # Save analysis report
    with open("cost_analysis_report.json", "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "models_tested": MODELS_TO_TEST,
            "cost_winner": analysis["cost_winner"],
            "accuracy_winner": analysis["accuracy_winner"],
            "best_tradeoff": analysis["best_tradeoff"],
            "detailed_results": results_by_model,
        }, f, indent=2)

    print(f"\n✅ Analysis saved to cost_analysis_report.json")

    # Recommendation
    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)

    cost_winner = analysis["cost_winner"]
    accuracy_winner = analysis["accuracy_winner"]
    tradeoff_winner = analysis["best_tradeoff"]

    if cost_winner:
        cost_data = results_by_model[cost_winner]["summary"]
        from costs import format_cost
        print(f"\n✅ Recommended grader model: {cost_winner}")
        print(f"   Cost: {format_cost(cost_data['avg_crag_cost_per_query'])}/query")
        print(f"   Hallucination rate: {cost_data['crag_hallucination_rate']}%")

        if cost_winner != "gpt-4o-mini-2024-07-18":
            savings = get_model_cost_ratio(cost_winner, "gpt-4o-mini-2024-07-18")
            savings_pct = (1 - savings) * 100
            print(f"   💰 Savings: {savings_pct:.0f}% cheaper than current gpt-4o-mini")

    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()

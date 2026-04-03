"""
costs.py
========
Cost tracking and analysis utilities for CRAG.

Provides:
  - Model pricing definitions
  - Cost calculation from token counts
  - Cost formatting utilities
  - Cumulative cost tracking per query/eval run
"""

from dataclasses import dataclass
from typing import Dict, Tuple


# Pricing per 1M tokens (input, output)
MODEL_PRICING: Dict[str, Tuple[float, float]] = {
    # Nano models (cost-optimized)
    "gpt-5-nano-2025-08-07": (0.05, 0.40),
    "gpt-4.1-nano-2025-04-14": (0.10, 0.40),

    # Standard models
    "gpt-4o-mini-2024-07-18": (0.15, 0.60),
    "gpt-4o": (5.00, 15.00),
    "gpt-4-turbo": (10.00, 30.00),
    "text-embedding-3-small": (0.02, 0.00),  # Embedding is input-only
}


def calculate_cost(model: str, input_tokens: int, output_tokens: int = 0) -> float:
    """
    Calculate cost in USD for an LLM API call.

    Args:
        model: Model identifier (must be in MODEL_PRICING)
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens (default 0 for embeddings)

    Returns:
        Cost in USD (float)

    Raises:
        ValueError: If model not found in pricing table
    """
    if model not in MODEL_PRICING:
        raise ValueError(f"Unknown model: {model}. Available: {list(MODEL_PRICING.keys())}")

    input_cost, output_cost = MODEL_PRICING[model]

    # Convert per-1M pricing to per-token
    input_price_per_token = input_cost / 1_000_000
    output_price_per_token = output_cost / 1_000_000

    total_cost = (input_tokens * input_price_per_token) + (output_tokens * output_price_per_token)

    return total_cost


def format_cost(cost_usd: float) -> str:
    """
    Format cost as human-readable string.

    Args:
        cost_usd: Cost in USD (float)

    Returns:
        Formatted string (e.g., "$0.0042", "$0.12", "$1.50")
    """
    if cost_usd < 0.01:
        return f"${cost_usd:.4f}"
    elif cost_usd < 1.00:
        return f"${cost_usd:.3f}"
    else:
        return f"${cost_usd:.2f}"


def get_model_cost_ratio(model1: str, model2: str) -> float:
    """
    Calculate cost ratio between two models.
    Useful for comparing, e.g., "gpt-5-nano is 3x cheaper than gpt-4o-mini".

    Assumes average 500 input + 200 output tokens per call.

    Args:
        model1: First model
        model2: Second model (baseline)

    Returns:
        Ratio (e.g., 0.33 means model1 is 33% of model2's cost, or "3x cheaper")
    """
    cost1 = calculate_cost(model1, 500, 200)
    cost2 = calculate_cost(model2, 500, 200)

    if cost2 == 0:
        return 1.0

    return cost1 / cost2


@dataclass
class CostBreakdown:
    """Track costs for a single LLM API call."""

    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float

    def __post_init__(self):
        """Validate and auto-calculate cost if not provided."""
        if self.cost_usd == 0:
            self.cost_usd = calculate_cost(self.model, self.input_tokens, self.output_tokens)

    def __str__(self) -> str:
        """Return human-readable cost breakdown."""
        return (
            f"{self.model}: {self.input_tokens} in + {self.output_tokens} out = {format_cost(self.cost_usd)}"
        )


@dataclass
class QueryCostSummary:
    """Aggregate costs for a complete query execution."""

    query: str
    total_cost_usd: float = 0.0
    cost_breakdown: list = None  # List[CostBreakdown]
    llm_calls: int = 0

    def __post_init__(self):
        if self.cost_breakdown is None:
            self.cost_breakdown = []

    def add_cost(self, breakdown: CostBreakdown):
        """Add a cost breakdown entry."""
        self.cost_breakdown.append(breakdown)
        self.total_cost_usd += breakdown.cost_usd
        self.llm_calls += 1

    def __str__(self) -> str:
        """Return human-readable summary."""
        return (
            f"Query: {self.query[:50]}...\n"
            f"  Total cost: {format_cost(self.total_cost_usd)}\n"
            f"  LLM calls: {self.llm_calls}\n"
            f"  Breakdown:\n"
            + "\n".join(f"    {cb}" for cb in self.cost_breakdown)
        )


class CostAnalysisReport:
    """Aggregate costs and metrics for an eval run across multiple models."""

    def __init__(self, models_tested: list):
        self.models_tested = models_tested
        self.results = {}  # model -> {hallucination_rate, cost_per_query, ...}

    def add_result(self, model: str, result_dict: dict):
        """Record evaluation results for a model."""
        self.results[model] = result_dict

    def get_cost_winner(self) -> str:
        """Return model with lowest cost per query."""
        if not self.results:
            return None

        return min(self.results.keys(),
                   key=lambda m: self.results[m].get('avg_cost_per_query', float('inf')))

    def get_accuracy_winner(self) -> str:
        """Return model with lowest hallucination rate."""
        if not self.results:
            return None

        return min(self.results.keys(),
                   key=lambda m: self.results[m].get('hallucination_rate', float('inf')))

    def get_best_tradeoff(self) -> Tuple[str, float]:
        """
        Return model that best balances cost and accuracy.
        Score = (cost / min_cost) + (hallucination_rate / max_hallucination_rate)
        Lower score is better.
        """
        if not self.results:
            return None, float('inf')

        min_cost = min(r.get('avg_cost_per_query', float('inf')) for r in self.results.values())
        max_hallucination = max(r.get('hallucination_rate', 0) for r in self.results.values())

        best_model = None
        best_score = float('inf')

        for model, result in self.results.items():
            cost_factor = result.get('avg_cost_per_query', 0) / min_cost if min_cost > 0 else 0
            hallucination_factor = (
                result.get('hallucination_rate', 0) / max_hallucination
                if max_hallucination > 0 else 0
            )
            score = cost_factor + hallucination_factor

            if score < best_score:
                best_score = score
                best_model = model

        return best_model, best_score

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = ["Cost Analysis Report", "=" * 50]

        for model, result in self.results.items():
            lines.append(f"\n{model}:")
            lines.append(f"  Hallucination rate: {result.get('hallucination_rate', 0):.1%}")
            lines.append(f"  Avg cost per query: {format_cost(result.get('avg_cost_per_query', 0))}")
            lines.append(f"  Total cost: {format_cost(result.get('total_cost', 0))}")

        cost_winner = self.get_cost_winner()
        accuracy_winner = self.get_accuracy_winner()
        tradeoff_winner, _ = self.get_best_tradeoff()

        lines.append(f"\n💰 Cheapest: {cost_winner}")
        lines.append(f"✅ Most accurate: {accuracy_winner}")
        lines.append(f"⚖️ Best tradeoff: {tradeoff_winner}")

        return "\n".join(lines)

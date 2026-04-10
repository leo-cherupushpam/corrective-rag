"""
test_costs.py
=============
Unit tests for the costs module (cost tracking and calculation).

Tests:
  - CostBreakdown calculation
  - calculate_cost() function
  - Token counting accuracy
  - Model pricing lookup
"""

import pytest
from costs import CostBreakdown, calculate_cost, MODEL_PRICING


@pytest.mark.unit
class TestCostBreakdown:
    """Tests for CostBreakdown dataclass."""

    def test_cost_breakdown_creation(self):
        """Test creating a cost breakdown."""
        cost = CostBreakdown(
            model="gpt-4o-mini-2024-07-18",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0,  # will be calculated
        )

        assert cost.model == "gpt-4o-mini-2024-07-18"
        assert cost.input_tokens == 100
        assert cost.output_tokens == 50
        assert cost.cost_usd > 0  # Should be calculated

    def test_cost_breakdown_pricing(self):
        """Test that cost is calculated correctly based on model."""
        # gpt-4o-mini pricing: input $0.15/1M, output $0.60/1M
        cost = CostBreakdown(
            model="gpt-4o-mini-2024-07-18",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            cost_usd=0,
        )

        # Input: 1M tokens * $0.15/1M = $0.15
        # Output: 1M tokens * $0.60/1M = $0.60
        # Total: $0.75
        assert cost.cost_usd == pytest.approx(0.75, rel=0.01)

    def test_cost_breakdown_small_amounts(self):
        """Test cost calculation for small token amounts."""
        cost = CostBreakdown(
            model="text-embedding-3-small",
            input_tokens=100,
            output_tokens=0,
            cost_usd=0,
        )

        # Embedding pricing: $0.02/1M tokens
        # 100 tokens * $0.02/1M = $0.000002
        assert cost.cost_usd > 0
        assert cost.cost_usd < 0.0001  # Should be very small

    def test_cost_breakdown_different_models(self):
        """Test that different models have correct pricing."""
        models_and_tokens = [
            ("gpt-5-nano-2025-08-07", 100, 100),
            ("gpt-4o-mini-2024-07-18", 100, 100),
            ("text-embedding-3-small", 100, 0),
        ]

        costs = []
        for model, input_tokens, output_tokens in models_and_tokens:
            cost = CostBreakdown(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=0,
            )
            costs.append((model, cost.cost_usd))

        # All should have a cost > 0
        for model, cost_usd in costs:
            assert cost_usd > 0, f"Model {model} should have positive cost"

    def test_cost_breakdown_zero_tokens(self):
        """Test cost breakdown with zero tokens."""
        cost = CostBreakdown(
            model="gpt-4o-mini-2024-07-18",
            input_tokens=0,
            output_tokens=0,
            cost_usd=0,
        )

        # Zero tokens should result in zero cost
        assert cost.cost_usd == 0


@pytest.mark.unit
class TestCalculateCost:
    """Tests for calculate_cost() function."""

    def test_calculate_cost_valid_model(self):
        """Test calculating cost for a valid model."""
        model = "gpt-4o-mini-2024-07-18"
        input_tokens = 1000
        output_tokens = 500

        cost = calculate_cost(model, input_tokens, output_tokens)

        assert cost > 0
        assert isinstance(cost, float)

    def test_calculate_cost_embedding_model(self):
        """Test cost calculation for embedding model."""
        model = "text-embedding-3-small"
        input_tokens = 1000

        cost = calculate_cost(model, input_tokens, 0)

        # Should be much cheaper than generation
        assert cost > 0
        assert cost < 0.001

    def test_calculate_cost_generation_expensive(self):
        """Test that generation is more expensive than embeddings."""
        embedding_cost = calculate_cost("text-embedding-3-small", 1000, 0)
        generation_cost = calculate_cost("gpt-4o-mini-2024-07-18", 1000, 1000)

        assert generation_cost > embedding_cost * 100

    def test_calculate_cost_invalid_model(self):
        """Test error handling for invalid model."""
        with pytest.raises(ValueError):
            calculate_cost("nonexistent-model-xyz", 100, 100)

    def test_calculate_cost_zero_tokens(self):
        """Test cost calculation with zero tokens."""
        cost = calculate_cost("gpt-4o-mini-2024-07-18", 0, 0)
        assert cost == 0


@pytest.mark.unit
class TestModelPricing:
    """Tests for MODEL_PRICING dictionary."""

    def test_model_pricing_has_generators(self):
        """Test that generator models are priced."""
        generators = [
            "gpt-5-nano-2025-08-07",
            "gpt-4o-mini-2024-07-18",
        ]

        for model in generators:
            assert model in MODEL_PRICING, f"Model {model} should be in pricing"
            assert MODEL_PRICING[model]["input"] > 0
            assert MODEL_PRICING[model]["output"] > 0

    def test_model_pricing_has_embeddings(self):
        """Test that embedding models are priced."""
        embeddings = ["text-embedding-3-small"]

        for model in embeddings:
            assert model in MODEL_PRICING
            assert MODEL_PRICING[model]["input"] > 0

    def test_model_pricing_ratio(self):
        """Test that output is more expensive than input (typical)."""
        # Most models: output tokens more expensive than input
        model = "gpt-4o-mini-2024-07-18"
        assert MODEL_PRICING[model]["output"] >= MODEL_PRICING[model]["input"]


@pytest.mark.unit
class TestCostAccuracy:
    """Tests for cost calculation accuracy."""

    def test_cumulative_costs(self):
        """Test summing multiple costs."""
        costs = [
            CostBreakdown("gpt-4o-mini-2024-07-18", 100, 50, 0),
            CostBreakdown("gpt-4o-mini-2024-07-18", 200, 100, 0),
            CostBreakdown("text-embedding-3-small", 500, 0, 0),
        ]

        total = sum(c.cost_usd for c in costs)
        assert total > 0

    def test_cost_per_query_example(self):
        """Test realistic cost per query calculation."""
        # Typical query: retrieve → grade → generate → verify
        retrieval_cost = CostBreakdown("text-embedding-3-small", 100, 0, 0)
        grading_costs = [
            CostBreakdown("gpt-4o-mini-2024-07-18", 200, 50, 0)
            for _ in range(3)  # 3 docs graded
        ]
        generation_cost = CostBreakdown("gpt-5-nano-2025-08-07", 1000, 200, 0)
        verification_cost = CostBreakdown("gpt-4o-mini-2024-07-18", 400, 100, 0)

        total_cost = (
            retrieval_cost.cost_usd
            + sum(c.cost_usd for c in grading_costs)
            + generation_cost.cost_usd
            + verification_cost.cost_usd
        )

        # Should be roughly $0.01-$0.03 per query based on model pricing
        assert 0.005 < total_cost < 0.05


@pytest.mark.parametrize("model", [
    "gpt-5-nano-2025-08-07",
    "gpt-4o-mini-2024-07-18",
    "text-embedding-3-small",
])
def test_cost_breakdown_all_models(model):
    """Parameterized test for cost breakdown across all models."""
    cost = CostBreakdown(model=model, input_tokens=100, output_tokens=50, cost_usd=0)

    assert cost.model == model
    assert cost.input_tokens == 100
    assert cost.output_tokens == 50
    assert cost.cost_usd >= 0

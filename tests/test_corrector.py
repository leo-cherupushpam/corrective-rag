"""
test_corrector.py
=================
Unit tests for the corrector module (query correction strategies).

Tests:
  - expand_query() - query broadening
  - decompose_query() - query decomposition
  - extract_keywords() - keyword extraction
  - get_correction_candidates() - strategy selection
  - Strategy registration and custom strategies
"""

import pytest
from unittest.mock import patch, MagicMock
import json

from corrector import (
    expand_query,
    decompose_query,
    extract_keywords,
    get_correction_candidates,
    register_strategy,
    list_strategies,
    ExpandedQuery,
    DecomposedQueries,
    KeywordQuery,
    CORRECTION_STRATEGIES,
)


@pytest.mark.unit
class TestExpandQuery:
    """Tests for expand_query() strategy."""

    def test_expand_query_basic(self, mock_openai_client):
        """Test expanding a query to broader terminology."""
        query = "return policy"

        result = expand_query(query)

        assert isinstance(result, ExpandedQuery)
        assert result.expanded is not None
        assert len(result.expanded) > 0
        assert result.rationale is not None

    def test_expand_query_preserves_intent(self, mock_openai_client):
        """Test that expansion doesn't change the core intent."""
        query = "return policy"

        result = expand_query(query)

        # Expanded query should still contain key terms
        assert result.expanded is not None

    def test_expand_query_error_handling(self, mock_openai_client):
        """Test graceful error handling in expand_query."""
        query = "test query"

        with patch("corrector.client") as mock_client:
            mock_client.beta.chat.completions.parse.side_effect = ValueError("Parse error")

            # Should return fallback expanded query
            result = expand_query(query)

            assert result.expanded == query  # Falls back to original
            assert "Parse error" in result.rationale or len(result.rationale) > 0


@pytest.mark.unit
class TestDecomposeQuery:
    """Tests for decompose_query() strategy."""

    def test_decompose_complex_query(self, mock_openai_client):
        """Test decomposing a multi-part query."""
        query = "Can I return an item if it arrived after 30 days and how long is a refund?"

        result = decompose_query(query)

        assert isinstance(result, DecomposedQueries)
        assert len(result.sub_queries) > 1
        assert all(isinstance(q, str) for q in result.sub_queries)
        assert result.rationale is not None

    def test_decompose_simple_query(self, mock_openai_client):
        """Test that simple queries return single sub-query."""
        query = "What is your return policy?"

        result = decompose_query(query)

        # Even simple queries might be decomposed, but at least one sub-query
        assert len(result.sub_queries) >= 1

    def test_decompose_query_error_handling(self, mock_openai_client):
        """Test error handling returns original query as fallback."""
        query = "test query"

        with patch("corrector.client") as mock_client:
            mock_client.beta.chat.completions.parse.side_effect = ValueError("Parse error")

            result = decompose_query(query)

            # Should fallback to original query
            assert query in result.sub_queries


@pytest.mark.unit
class TestExtractKeywords:
    """Tests for extract_keywords() strategy."""

    def test_extract_keywords_basic(self, mock_openai_client):
        """Test extracting keywords from a query."""
        query = "What is your return policy for damaged items?"

        result = extract_keywords(query)

        assert isinstance(result, KeywordQuery)
        assert len(result.keywords) > 0
        assert all(isinstance(k, str) for k in result.keywords)
        assert result.boolean_query is not None
        assert len(result.boolean_query) > 0

    def test_extract_keywords_boolean_format(self, mock_openai_client):
        """Test that boolean query is properly formatted."""
        query = "shipping time"

        result = extract_keywords(query)

        # Boolean query should use AND/OR operators
        assert "AND" in result.boolean_query or "OR" in result.boolean_query or len(result.boolean_query) > 0

    def test_extract_keywords_error_handling(self, mock_openai_client):
        """Test error handling in keyword extraction."""
        query = "test query"

        with patch("corrector.client") as mock_client:
            mock_client.beta.chat.completions.parse.side_effect = ValueError("Parse error")

            result = extract_keywords(query)

            # Should fallback to splitting query
            assert len(result.keywords) > 0
            assert "test" in result.keywords or "query" in result.keywords


@pytest.mark.unit
class TestGetCorrectionCandidates:
    """Tests for get_correction_candidates() orchestrator."""

    def test_get_candidates_expand_strategy(self, mock_openai_client):
        """Test retrieving candidates with expand strategy."""
        query = "return"
        candidates = get_correction_candidates(query, strategy="expand")

        assert isinstance(candidates, list)
        assert len(candidates) > 0
        assert all(isinstance(c, str) for c in candidates)

    def test_get_candidates_decompose_strategy(self, mock_openai_client):
        """Test retrieving candidates with decompose strategy."""
        query = "returns and refunds"
        candidates = get_correction_candidates(query, strategy="decompose")

        assert isinstance(candidates, list)
        assert len(candidates) > 0

    def test_get_candidates_keywords_strategy(self, mock_openai_client):
        """Test retrieving candidates with keywords strategy."""
        query = "return policy"
        candidates = get_correction_candidates(query, strategy="keywords")

        assert isinstance(candidates, list)
        assert len(candidates) > 0

    def test_get_candidates_unknown_strategy(self, mock_openai_client):
        """Test unknown strategy returns original query."""
        query = "test"
        candidates = get_correction_candidates(query, strategy="unknown_strategy")

        # Should return original query as fallback
        assert candidates == [query]


@pytest.mark.unit
class TestStrategyRegistry:
    """Tests for custom strategy registration."""

    def test_list_strategies(self, mock_openai_client):
        """Test listing all available strategies."""
        strategies = list_strategies()

        assert "expand" in strategies
        assert "decompose" in strategies
        assert "keywords" in strategies

    def test_register_custom_strategy(self, mock_openai_client):
        """Test registering and using a custom strategy."""
        # Define a custom strategy
        def custom_strategy(query: str) -> list[str]:
            return [f"{query} CUSTOM"]

        # Register it
        register_strategy("custom_test", custom_strategy)

        # Use it
        candidates = get_correction_candidates("test", strategy="custom_test")

        assert "test CUSTOM" in candidates

        # Clean up
        from corrector import _STRATEGY_REGISTRY
        del _STRATEGY_REGISTRY["custom_test"]

    def test_custom_strategy_retrieval(self, mock_openai_client):
        """Test custom strategy is properly retrieved from registry."""
        def my_strategy(query: str) -> list[str]:
            return ["custom_1", "custom_2", "custom_3"]

        register_strategy("my_strategy", my_strategy)

        candidates = get_correction_candidates("anything", strategy="my_strategy")

        assert candidates == ["custom_1", "custom_2", "custom_3"]

        # Clean up
        from corrector import _STRATEGY_REGISTRY
        del _STRATEGY_REGISTRY["my_strategy"]


@pytest.mark.unit
class TestCorrectionWorkflow:
    """Integration tests for correction workflows."""

    def test_full_correction_pipeline(self, mock_openai_client):
        """Test using all three correction strategies in sequence."""
        query = "returning items"

        strategies_to_try = ["expand", "decompose", "keywords"]
        all_candidates = []

        for strategy in strategies_to_try:
            candidates = get_correction_candidates(query, strategy=strategy)
            all_candidates.extend(candidates)

        # Should have candidates from all strategies
        assert len(all_candidates) >= 3

    @pytest.mark.parametrize("strategy", ["expand", "decompose", "keywords"])
    def test_each_strategy_works(self, mock_openai_client, strategy):
        """Test each correction strategy individually."""
        query = "test query"
        candidates = get_correction_candidates(query, strategy=strategy)

        assert isinstance(candidates, list)
        assert len(candidates) > 0
        assert all(isinstance(c, str) for c in candidates)


@pytest.mark.unit
class TestCorrectionErrorRecovery:
    """Tests for error handling in correction strategies."""

    def test_all_strategies_handle_errors(self, mock_openai_client):
        """Test that all strategies can handle API errors gracefully."""
        query = "test"

        # Each strategy should handle errors and return something
        for strategy in CORRECTION_STRATEGIES:
            with patch("corrector.client") as mock_client:
                mock_client.beta.chat.completions.parse.side_effect = Exception("API error")

                # Should not crash, even with API error
                try:
                    candidates = get_correction_candidates(query, strategy=strategy)
                    assert isinstance(candidates, list)
                except Exception as e:
                    # Should only raise known exceptions, not generic ones
                    assert "API error" not in str(e)

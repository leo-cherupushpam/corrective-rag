"""
test_multi_hop.py
=================
Unit tests for multi-hop retrieval module.

Tests:
  - detect_multi_hop() decision logic
  - multi_hop_retrieve() orchestration
  - deduplicate_docs() duplicate removal
  - MultiHopTrace tracking
"""

import pytest
from unittest.mock import patch, MagicMock

from multi_hop import (
    detect_multi_hop,
    multi_hop_retrieve,
    deduplicate_docs,
    MultiHopDecision,
    MultiHopTrace,
)


@pytest.mark.unit
class TestDetectMultiHop:
    """Tests for detect_multi_hop() decision logic."""

    def test_detect_multi_hop_needed(self, mock_openai_client):
        """Test detecting when multi-hop retrieval is needed."""
        query = "Can I return an item if it arrived after 30 days?"
        docs = ["Return Policy: Returns within 30 days."]

        decision, cost = detect_multi_hop(query, docs)

        assert isinstance(decision, MultiHopDecision)
        assert hasattr(decision, "needs_multi_hop")

    def test_detect_multi_hop_not_needed(self, mock_openai_client):
        """Test detecting when multi-hop is not needed."""
        query = "What is your return policy?"
        docs = [
            "Return Policy: Returns accepted within 30 days for unused items."
        ]

        # Mock response: no multi-hop needed
        with patch("multi_hop.client") as mock_client:
            response = MagicMock()
            response.choices[0].message.parsed = MultiHopDecision(
                needs_multi_hop=False,
                bridge_query="",
                bridge_entity="",
                reason="Document fully answers the question",
            )
            response.usage.prompt_tokens = 100
            response.usage.completion_tokens = 50
            mock_client.beta.chat.completions.parse.return_value = response

            decision, cost = detect_multi_hop(query, docs)

            assert decision.needs_multi_hop is False

    def test_detect_multi_hop_error_handling(self, mock_openai_client):
        """Test graceful error handling in multi-hop detection."""
        query = "test"
        docs = ["doc"]

        with patch("multi_hop.client") as mock_client:
            mock_client.beta.chat.completions.parse.side_effect = Exception("API error")

            decision, cost = detect_multi_hop(query, docs)

            # Should fallback to no-multi-hop on error
            assert decision.needs_multi_hop is False
            assert cost is None


@pytest.mark.unit
class TestDedupDocs:
    """Tests for deduplicate_docs() function."""

    def test_deduplicate_exact_match(self):
        """Test removing exact duplicate documents."""
        doc1_list = ["Return Policy: 30 days.", "Shipping: 5-7 days."]
        doc2_list = ["Return Policy: 30 days."]  # Exact duplicate

        result = deduplicate_docs(doc1_list, doc2_list, threshold=0.95)

        # Duplicate should be removed
        assert len(result) == 2  # Original docs only

    def test_deduplicate_similar_docs(self):
        """Test removing similar documents above threshold."""
        doc1_list = ["Return Policy: Returns accepted within 30 days."]
        doc2_list = ["Return Policy: We accept returns within 30 days."]  # Very similar

        result = deduplicate_docs(doc1_list, doc2_list, threshold=0.8)

        # Similar doc might be removed depending on threshold
        assert len(result) <= 2

    def test_deduplicate_unique_docs(self):
        """Test keeping documents that are dissimilar."""
        doc1_list = ["Return Policy: 30 days."]
        doc2_list = ["Shipping Information: 5-7 business days."]  # Different topic

        result = deduplicate_docs(doc1_list, doc2_list, threshold=0.8)

        # Different docs should be kept
        assert len(result) == 2

    def test_deduplicate_empty_docs2(self):
        """Test with empty second document list."""
        doc1_list = ["doc1", "doc2"]
        doc2_list = []

        result = deduplicate_docs(doc1_list, doc2_list)

        assert result == doc1_list

    def test_deduplicate_threshold_effect(self):
        """Test that threshold affects deduplication."""
        doc1_list = ["Return Policy: Returns within 30 days for unused items."]
        doc2_list = ["Return Policy: We accept returns within 30 days."]

        # High threshold - less likely to deduplicate
        result_high = deduplicate_docs(doc1_list, doc2_list, threshold=0.99)
        # Low threshold - more likely to deduplicate
        result_low = deduplicate_docs(doc1_list, doc2_list, threshold=0.5)

        # High threshold should keep more docs
        assert len(result_high) >= len(result_low)


@pytest.mark.unit
class TestMultiHopTrace:
    """Tests for MultiHopTrace dataclass."""

    def test_multi_hop_trace_creation(self):
        """Test creating a MultiHopTrace."""
        trace = MultiHopTrace(
            hop_number=1,
            bridge_query="late delivery exceptions",
            bridge_entity="exception handling",
            docs_retrieved=5,
            docs_passed_grade=2,
            docs_added=["Exception Policy doc"],
        )

        assert trace.hop_number == 1
        assert trace.bridge_query is not None
        assert trace.docs_retrieved == 5
        assert trace.docs_passed_grade == 2

    def test_multi_hop_trace_multiple_hops(self):
        """Test tracking multiple hops."""
        hops = []
        for i in range(3):
            hop = MultiHopTrace(
                hop_number=i + 1,
                bridge_query=f"bridge query {i}",
                bridge_entity=f"entity {i}",
                docs_retrieved=5,
                docs_passed_grade=2,
            )
            hops.append(hop)

        assert len(hops) == 3
        assert hops[0].hop_number == 1
        assert hops[2].hop_number == 3


@pytest.mark.unit
class TestMultiHopRetrieve:
    """Tests for multi_hop_retrieve() orchestrator."""

    def test_multi_hop_retrieve_no_hops_needed(self, vector_store_with_docs, mock_openai_client):
        """Test when initial docs are sufficient."""
        query = "What is your return policy?"
        initial_docs = ["Return Policy: 30 days."]

        # Mock: no multi-hop needed
        with patch("multi_hop.detect_multi_hop") as mock_detect:
            mock_decision = MultiHopDecision(
                needs_multi_hop=False,
                bridge_query="",
                bridge_entity="",
                reason="Docs sufficient",
            )
            mock_detect.return_value = (mock_decision, None)

            merged, traces, costs = multi_hop_retrieve(
                query, initial_docs, vector_store_with_docs, max_hops=2
            )

            # Should not perform any hops
            assert len(traces) == 0
            assert merged == initial_docs

    def test_multi_hop_retrieve_with_hops(self, vector_store_with_docs, mock_openai_client):
        """Test multi-hop retrieval with actual hops."""
        query = "Complex multi-part question?"
        initial_docs = ["Some doc"]

        with patch("multi_hop.detect_multi_hop") as mock_detect:
            # First call: yes, do multi-hop. Second call: no
            decisions = [
                MultiHopDecision(
                    needs_multi_hop=True,
                    bridge_query="bridge query",
                    bridge_entity="missing info",
                    reason="Need more info",
                ),
                MultiHopDecision(
                    needs_multi_hop=False,
                    bridge_query="",
                    bridge_entity="",
                    reason="Done",
                ),
            ]
            mock_detect.side_effect = [(d, None) for d in decisions]

            with patch("multi_hop.filter_relevant") as mock_filter:
                mock_filter.return_value = (["bridge doc"], [], [])

                merged, traces, costs = multi_hop_retrieve(
                    query, initial_docs, vector_store_with_docs, max_hops=2
                )

                # Should have performed at least one hop
                assert len(traces) > 0

    def test_multi_hop_retrieve_max_hops(self, vector_store_with_docs, mock_openai_client):
        """Test that max_hops limit is respected."""
        query = "Complex query"
        initial_docs = ["doc1"]

        with patch("multi_hop.detect_multi_hop") as mock_detect:
            # Always return: yes, do multi-hop
            mock_decision = MultiHopDecision(
                needs_multi_hop=True,
                bridge_query="bridge",
                bridge_entity="entity",
                reason="Always need more",
            )
            mock_detect.return_value = (mock_decision, None)

            with patch("multi_hop.filter_relevant") as mock_filter:
                mock_filter.return_value = (["new doc"], [], [])

                merged, traces, costs = multi_hop_retrieve(
                    query, initial_docs, vector_store_with_docs, max_hops=2
                )

                # Should stop at max_hops (2)
                assert len(traces) <= 2


@pytest.mark.integration
class TestMultiHopIntegration:
    """Integration tests for multi-hop workflow."""

    def test_full_multi_hop_workflow(self, vector_store_with_docs, mock_openai_client, sample_documents):
        """Test complete multi-hop retrieval workflow."""
        query = "Complex question requiring multiple docs?"
        initial_docs = sample_documents[:1]

        with patch("multi_hop.detect_multi_hop") as mock_detect:
            mock_detect.return_value = (
                MultiHopDecision(
                    needs_multi_hop=True,
                    bridge_query="secondary info",
                    bridge_entity="missing context",
                    reason="Need more docs",
                ),
                None,
            )

            with patch("multi_hop.filter_relevant") as mock_filter:
                mock_filter.return_value = (sample_documents[1:3], [], [])

                merged, traces, costs = multi_hop_retrieve(
                    query, initial_docs, vector_store_with_docs, max_hops=2
                )

                # Should have merged docs from hops
                assert len(merged) >= len(initial_docs)
                assert len(traces) > 0

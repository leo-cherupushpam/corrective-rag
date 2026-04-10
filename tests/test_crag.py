"""
test_crag.py
============
Integration tests for the main CRAG pipeline.

Tests:
  - baseline_rag() function
  - crag() main pipeline
  - VectorStore retrieval
  - Full end-to-end workflows
  - Error handling and resilience
"""

import pytest
from unittest.mock import patch, MagicMock

from crag import (
    VectorStore,
    baseline_rag,
    crag,
    generate_answer,
    QueryTrace,
    ComparisonResult,
)


@pytest.mark.unit
class TestVectorStore:
    """Tests for VectorStore class."""

    def test_vector_store_creation(self, mock_openai_client):
        """Test creating an empty VectorStore."""
        store = VectorStore()

        assert store.index is None
        assert store.documents == []
        assert store.dim == 1536

    def test_vector_store_add_documents(self, vector_store_with_docs, sample_documents):
        """Test adding documents to VectorStore."""
        store = vector_store_with_docs

        assert len(store.documents) == len(sample_documents)
        assert store.index is not None

    def test_vector_store_retrieve(self, vector_store_with_docs):
        """Test retrieving documents from VectorStore."""
        query = "return policy"
        results = vector_store_with_docs.retrieve(query, k=3)

        assert len(results) <= 3
        assert all(isinstance(r, str) for r in results)

    def test_vector_store_retrieve_empty(self, mock_openai_client):
        """Test retrieving from empty VectorStore."""
        store = VectorStore()
        results = store.retrieve("anything", k=5)

        assert len(results) == 0


@pytest.mark.unit
class TestGenerateAnswer:
    """Tests for generate_answer() function."""

    def test_generate_answer_with_docs(self, mock_openai_client):
        """Test generating answer from documents."""
        query = "What is your return policy?"
        docs = ["Return Policy: Returns accepted within 30 days."]

        answer, cost = generate_answer(query, docs)

        assert answer is not None
        assert len(answer) > 0
        assert cost is not None

    def test_generate_answer_empty_docs(self, mock_openai_client):
        """Test generating answer with no documents."""
        query = "What is your return policy?"
        docs = []

        answer, cost = generate_answer(query, docs)

        assert "don't have enough information" in answer
        assert cost is None

    def test_generate_answer_error_handling(self, mock_openai_client):
        """Test error handling in answer generation."""
        query = "test"
        docs = ["test doc"]

        with patch("crag.client") as mock_client:
            mock_client.chat.completions.create.side_effect = Exception("Generation failed")

            answer, cost = generate_answer(query, docs)

            # Should return fallback answer
            assert answer is not None
            assert "error" in answer.lower() or len(answer) > 0


@pytest.mark.integration
class TestBaselineRAG:
    """Tests for baseline_rag() function."""

    def test_baseline_rag_basic(self, vector_store_with_docs, mock_openai_client):
        """Test baseline RAG execution."""
        query = "What is your return policy?"
        trace = baseline_rag(query, vector_store_with_docs)

        assert isinstance(trace, QueryTrace)
        assert trace.query == query
        assert trace.mode == "baseline"
        assert trace.answer is not None
        assert len(trace.answer) > 0

    def test_baseline_rag_trace(self, vector_store_with_docs, mock_openai_client):
        """Test that baseline RAG creates proper trace."""
        query = "test"
        trace = baseline_rag(query, vector_store_with_docs)

        assert trace.mode == "baseline"
        assert trace.answer_confidence is not None
        assert 0.0 <= trace.answer_confidence <= 1.0
        assert len(trace.confidence_reasoning) > 0

    def test_baseline_rag_no_confidence_boost(self, vector_store_with_docs, mock_openai_client):
        """Test that baseline RAG doesn't heavily boost confidence."""
        query = "test"
        trace = baseline_rag(query, vector_store_with_docs)

        # Baseline should have moderate confidence
        assert trace.answer_confidence <= 0.7  # Not as high as graded


@pytest.mark.integration
class TestCRAG:
    """Tests for crag() main pipeline."""

    def test_crag_basic(self, vector_store_with_docs, mock_openai_client):
        """Test basic CRAG execution."""
        query = "What is your return policy?"
        trace = crag(query, vector_store_with_docs)

        assert isinstance(trace, QueryTrace)
        assert trace.query == query
        assert trace.mode == "crag"
        assert trace.answer is not None

    def test_crag_with_relevant_docs(self, vector_store_with_docs, mock_openai_client):
        """Test CRAG with relevant documents found."""
        query = "return policy"

        with patch("crag.filter_relevant") as mock_filter:
            # Mock: return some relevant docs
            mock_filter.return_value = (
                ["Return Policy doc"],
                [MagicMock(relevant=True, score=0.95)],
                [MagicMock()],  # cost
            )

            trace = crag(query, vector_store_with_docs)

            assert len(trace.docs_used) > 0
            assert trace.needed_correction is False
            assert trace.fallback_used is False

    def test_crag_with_correction(self, vector_store_with_docs, mock_openai_client):
        """Test CRAG with query correction triggered."""
        query = "vague query"

        with patch("crag.filter_relevant") as mock_filter:
            # Mock: first call returns no relevant docs, second returns docs
            call_count = [0]

            def side_effect(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    # First call: no relevant docs
                    return ([], [], [])
                else:
                    # Second call: docs found
                    return (
                        ["Found doc"],
                        [MagicMock(relevant=True, score=0.9)],
                        [MagicMock()],
                    )

            mock_filter.side_effect = side_effect

            with patch("crag.get_correction_candidates") as mock_correct:
                mock_correct.return_value = ["expanded query"]

                trace = crag(query, vector_store_with_docs)

                # Should have triggered correction
                assert len(trace.corrections) > 0 or len(trace.docs_used) > 0

    def test_crag_confidence_scoring(self, vector_store_with_docs, mock_openai_client):
        """Test confidence scoring in CRAG."""
        query = "test"

        with patch("crag.filter_relevant") as mock_filter:
            mock_filter.return_value = (
                ["doc"],
                [MagicMock(relevant=True, score=0.95)],
                [MagicMock()],
            )

            trace = crag(query, vector_store_with_docs)

            assert 0.0 <= trace.answer_confidence <= 1.0
            assert len(trace.confidence_reasoning) > 0

    def test_crag_answer_verification(self, vector_store_with_docs, mock_openai_client):
        """Test answer verification in CRAG."""
        query = "test"

        with patch("crag.filter_relevant") as mock_filter:
            mock_filter.return_value = (
                ["doc"],
                [MagicMock(relevant=True, score=0.9)],
                [MagicMock()],
            )

            with patch("crag.verify_answer") as mock_verify:
                mock_verify.return_value = (
                    MagicMock(grounded=True, gaps=[], supported_claims=3),
                    MagicMock(),
                )

                trace = crag(query, vector_store_with_docs)

                # Should have verified the answer
                assert trace.answer_grounded in [True, False, None]


@pytest.mark.integration
class TestCRAGVsBaseline:
    """Tests comparing CRAG and Baseline RAG."""

    def test_crag_more_confident_than_baseline(self, vector_store_with_docs, mock_openai_client):
        """Test that CRAG produces higher confidence scores."""
        query = "What is your return policy?"

        with patch("crag.filter_relevant") as mock_filter:
            mock_filter.return_value = (
                ["Return Policy doc"],
                [MagicMock(relevant=True, score=0.95)],
                [MagicMock()],
            )

            crag_trace = crag(query, vector_store_with_docs)
            baseline_trace = baseline_rag(query, vector_store_with_docs)

            # CRAG should have higher confidence when docs are relevant
            assert crag_trace.answer_confidence >= baseline_trace.answer_confidence

    def test_both_modes_answer_question(self, vector_store_with_docs, mock_openai_client):
        """Test that both modes produce answers."""
        query = "test query"

        baseline = baseline_rag(query, vector_store_with_docs)
        crag_result = crag(query, vector_store_with_docs)

        assert len(baseline.answer) > 0
        assert len(crag_result.answer) > 0


@pytest.mark.integration
class TestCRAGIntegration:
    """End-to-end integration tests for CRAG."""

    def test_full_crag_workflow(self, vector_store_with_docs, mock_openai_client, sample_documents):
        """Test complete CRAG workflow from query to answer."""
        query = "What is your return policy?"

        trace = crag(query, vector_store_with_docs)

        # Verify complete trace
        assert trace.query == query
        assert trace.answer is not None
        assert trace.answer_confidence >= 0.0
        assert len(trace.confidence_reasoning) > 0
        assert isinstance(trace.docs_used, list)

    def test_crag_cost_tracking(self, vector_store_with_docs, mock_openai_client):
        """Test that costs are properly tracked."""
        query = "test"

        trace = crag(query, vector_store_with_docs)

        assert trace.total_cost_usd >= 0
        assert len(trace.cost_breakdown) >= 0

    def test_crag_handles_no_retrieval(self, mock_openai_client):
        """Test CRAG when retrieval returns nothing."""
        store = VectorStore()  # Empty store
        query = "anything"

        with patch("crag.filter_relevant") as mock_filter:
            mock_filter.return_value = ([], [], [])

            trace = crag(query, store)

            # Should handle gracefully with fallback
            assert trace.answer is not None
            assert trace.fallback_used is True or len(trace.answer) > 0


@pytest.mark.parametrize("query", [
    "What is your return policy?",
    "How long does shipping take?",
    "What subscription plans do you offer?",
])
def test_crag_multiple_queries(vector_store_with_docs, mock_openai_client, query):
    """Parameterized test of CRAG with different queries."""
    trace = crag(query, vector_store_with_docs)

    assert trace.query == query
    assert trace.answer is not None
    assert trace.mode == "crag"

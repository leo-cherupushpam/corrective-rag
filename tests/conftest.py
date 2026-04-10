"""
conftest.py
===========
Pytest configuration and fixtures for CRAG tests.

Provides:
  - Mocked OpenAI API client (no live API calls)
  - Sample documents and queries
  - Mock API responses for all grader, generator, and corrector calls
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add app directory to path so we can import CRAG modules
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))


# ============================================================================
# OpenAI API Mocking
# ============================================================================


class MockChatCompletion:
    """Mock response from OpenAI chat.completions API."""

    def __init__(self, content: str, model: str = "gpt-4o-mini"):
        self.choices = [MagicMock()]
        self.choices[0].message.content = content
        self.choices[0].message.parsed = json.loads(content) if "{" in content else None
        self.usage = MagicMock()
        self.usage.prompt_tokens = 100
        self.usage.completion_tokens = 50
        self.model = model


class MockEmbedding:
    """Mock response from OpenAI embeddings API."""

    def __init__(self, vector_size: int = 1536, num_vectors: int = 1):
        self.data = [MagicMock() for _ in range(num_vectors)]
        # Each embedding is a list of floats (size 1536)
        for item in self.data:
            item.embedding = [0.1 * i for i in range(vector_size)]


@pytest.fixture
def mock_openai_client(monkeypatch):
    """
    Mock OpenAI client for all tests.

    Patches the OpenAI client at module level so all imports get the mocked version.
    """
    import openai

    original_client = openai.OpenAI

    def mock_openai_init(*args, **kwargs):
        """Create a mock OpenAI client."""
        mock_client = MagicMock()

        # Mock embeddings.create()
        def mock_embeddings_create(model, input):
            if isinstance(input, str):
                return MockEmbedding(num_vectors=1)
            return MockEmbedding(num_vectors=len(input))

        mock_client.embeddings.create = mock_embeddings_create

        # Mock chat.completions.create()
        def mock_chat_create(**kwargs):
            model = kwargs.get("model", "gpt-4o")
            # Return basic response for generation
            return MockChatCompletion("This is a generated answer.", model=model)

        mock_client.chat.completions.create = mock_chat_create

        # Mock beta.chat.completions.parse() for structured outputs
        def mock_beta_parse(**kwargs):
            response_format = kwargs.get("response_format")
            model = kwargs.get("model", "gpt-4o-mini")

            # Return appropriate mock based on response format
            if response_format.__name__ == "GradeResult":
                content = json.dumps({"relevant": True, "score": 0.9, "reason": "Test grade"})
            elif response_format.__name__ == "AnswerVerification":
                content = json.dumps({
                    "grounded": True,
                    "confidence": 0.95,
                    "gaps": [],
                    "supported_claims": 3,
                })
            elif response_format.__name__ == "MultiHopDecision":
                content = json.dumps({
                    "needs_multi_hop": False,
                    "bridge_query": "",
                    "bridge_entity": "",
                    "reason": "Documents sufficient",
                })
            elif response_format.__name__ == "ExpandedQuery":
                content = json.dumps({
                    "expanded": "expanded query version",
                    "rationale": "Broadening search",
                })
            elif response_format.__name__ == "DecomposedQueries":
                content = json.dumps({
                    "sub_queries": ["sub query 1", "sub query 2"],
                    "rationale": "Breaking into parts",
                })
            elif response_format.__name__ == "KeywordQuery":
                content = json.dumps({
                    "keywords": ["keyword1", "keyword2", "keyword3"],
                    "boolean_query": "keyword1 AND keyword2",
                })
            else:
                content = json.dumps({})

            response = MockChatCompletion(content, model=model)
            # Parse the JSON into the expected format
            try:
                response.choices[0].message.parsed = response_format(**json.loads(content))
            except:
                pass
            return response

        mock_client.beta.chat.completions.parse = mock_beta_parse

        return mock_client

    # Patch the OpenAI constructor
    monkeypatch.setattr(openai, "OpenAI", mock_openai_init)

    # Return a mock client instance
    return mock_openai_init()


# ============================================================================
# Sample Data Fixtures
# ============================================================================


@pytest.fixture
def sample_documents():
    """Sample FAQ documents for testing."""
    return [
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


@pytest.fixture
def sample_queries():
    """Sample queries for testing."""
    return {
        "direct": [
            "What is your return policy?",
            "How long does shipping take?",
            "What subscription plans do you offer?",
        ],
        "inference": [
            "Can I return a digital product?",
            "If I order today, when should I expect my package?",
            "How long is my data kept after I cancel?",
        ],
        "unanswerable": [
            "Do you accept returns after 30 days?",
            "What's your OAuth provider?",
            "Do you have a physical store location?",
        ],
    }


@pytest.fixture
def sample_query_trace():
    """Sample QueryTrace for testing."""
    from crag import QueryTrace, GradeTrace

    trace = QueryTrace(
        query="What is your return policy?",
        mode="crag",
        answer="Returns accepted within 30 days.",
        docs_used=["Return Policy doc"],
        grades=[
            GradeTrace(
                document_preview="Return Policy: We accept returns within 30 days...",
                relevant=True,
                score=0.95,
                reason="Directly answers the query",
            )
        ],
        answer_confidence=0.9,
        confidence_reasoning="High relevance, well-grounded answer.",
    )
    return trace


# ============================================================================
# Vector Store Fixture
# ============================================================================


@pytest.fixture
def vector_store_with_docs(sample_documents, mock_openai_client):
    """Create a VectorStore with sample documents pre-indexed."""
    from crag import VectorStore

    store = VectorStore()
    # Patch the embedding to return dummy vectors
    store.add_documents(sample_documents)
    return store


# ============================================================================
# Configuration
# ============================================================================


def pytest_configure(config):
    """Configure pytest plugins and options."""
    # Register custom markers
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test (fast, no API calls)"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test (uses multiple modules)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow (takes >1 second)"
    )

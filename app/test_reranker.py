#!/usr/bin/env python3
"""
Quick test of the reranker module.
Tests basic functionality before integrating into full pipeline.
"""

import sys
from reranker import score_documents, rerank_documents, should_rerank, get_reranking_info

# Test documents
test_docs = [
    "Python is a programming language used for web development, data science, and automation.",
    "The Great Wall of China is one of the most iconic landmarks in the world.",
    "Machine learning is a subset of artificial intelligence focused on data-driven prediction.",
    "Coffee is a beverage made from roasted coffee beans from the Coffea plant.",
    "Natural language processing enables computers to understand and generate human language.",
]

test_query = "What is machine learning and how does it relate to AI?"

print("=" * 60)
print("Testing Reranker Module")
print("=" * 60)

print("\n1. Testing should_rerank():")
print(f"   Documents: {len(test_docs)}")
print(f"   should_rerank: {should_rerank(test_docs)}")

print("\n2. Testing score_documents():")
print(f"   Query: '{test_query}'")
print()

try:
    results = score_documents(test_query, test_docs, k=3)

    for result in results:
        print(f"   Rank {result.rank + 1}: Score {result.score:.4f}")
        print(f"   Preview: {result.preview}")
        print()

    print("3. Testing get_reranking_info():")
    info = get_reranking_info(results)
    print(f"   Top score: {info['top_score']:.4f}")
    print(f"   Avg score: {info['avg_score']:.4f}")
    print(f"   Min score: {info['min_score']:.4f}")
    print(f"   Docs count: {info['docs_count']}")

    print("\n4. Testing rerank_documents() (convenience wrapper):")
    reranked = rerank_documents(test_query, test_docs, k=3)
    print(f"   Input docs: {len(test_docs)}, Output docs: {len(reranked)}")
    print(f"   Top doc preview: {reranked[0][:60]}...")

    print("\n✅ All tests passed!")
    sys.exit(0)

except Exception as e:
    print(f"\n❌ Test failed with error:")
    print(f"   {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

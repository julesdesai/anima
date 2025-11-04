#!/usr/bin/env python3
"""Inspect vector database contents and test search"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.vector_db import VectorDatabase
from src.corpus.embed import EmbeddingGenerator
from src.config import get_config

def main():
    config = get_config()
    db = VectorDatabase(config)
    embedder = EmbeddingGenerator(config)

    print("=" * 70)
    print("VECTOR DATABASE INSPECTION")
    print("=" * 70)

    # Get collection info
    info = db.get_collection_info()
    print(f"\nCollection: {config.vector_db.collection_name}")
    print(f"Total documents: {info.get('points_count', 0)}")
    print(f"Vector count: {info.get('vectors_count', 0)}")

    # Test search
    print("\n" + "=" * 70)
    print("TEST SEARCH")
    print("=" * 70)

    test_queries = [
        "digital minds ethics",
        "AI safety",
        "email communication",
        "projects working on",
    ]

    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 70)

        # Generate embedding
        query_emb = embedder.generate_one(query)

        # Search
        results = db.search(query_vector=query_emb, k=3)

        print(f"Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"\n  Result {i}:")
            print(f"    Similarity: {result.similarity:.3f}")
            print(f"    Source: {result.metadata.get('source', 'unknown')}")
            print(f"    File: {result.metadata.get('file_path', 'unknown')}")
            print(f"    Preview: {result.text[:150]}...")

    print("\n" + "=" * 70)
    print("SAMPLE DOCUMENTS")
    print("=" * 70)

    # Get a few random documents
    print("\nSearching for 'the' to get random samples...")
    query_emb = embedder.generate_one("the")
    samples = db.search(query_vector=query_emb, k=5)

    for i, doc in enumerate(samples, 1):
        print(f"\n📄 Document {i}:")
        print(f"   Source: {doc.metadata.get('source')}")
        print(f"   File: {doc.metadata.get('file_path', 'unknown')}")
        print(f"   Length: {len(doc.text)} chars")
        print(f"   Content: {doc.text[:200]}...")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()

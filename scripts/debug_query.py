#!/usr/bin/env python3
"""Debug a specific query to see what the agent searches for and finds"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.factory import AgentFactory
from src.config import get_config
import logging

# Detailed logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

def main():
    parser = argparse.ArgumentParser(description="Debug agent query")
    parser.add_argument("query", type=str, help="Query to debug")
    args = parser.parse_args()

    config = get_config()
    agent = AgentFactory.create_primary(config.user.name, config)

    print("\n" + "=" * 70)
    print(f"DEBUGGING QUERY: {args.query}")
    print("=" * 70)

    result = agent.respond(args.query)

    print("\n" + "=" * 70)
    print("TOOL CALLS MADE:")
    print("=" * 70)

    if not result['tool_calls']:
        print("⚠️  NO TOOL CALLS MADE!")
    else:
        for i, call in enumerate(result['tool_calls'], 1):
            print(f"\nTool Call {i}:")
            print(f"  Tool: {call['tool']}")
            print(f"  Query: {call['input'].get('query', 'N/A')}")
            print(f"  K: {call['input'].get('k', 'default')}")
            print(f"  Results returned: {call['result_count']}")

    print("\n" + "=" * 70)
    print("AGENT RESPONSE:")
    print("=" * 70)
    print(result['response'])

    print("\n" + "=" * 70)
    print(f"Total iterations: {result['iterations']}")
    print(f"Total tool calls: {len(result['tool_calls'])}")
    print("=" * 70)

    # Now manually test what SHOULD have been found
    print("\n" + "=" * 70)
    print("MANUAL SEARCH TEST:")
    print("=" * 70)

    from src.corpus.embed import EmbeddingGenerator
    from src.database.vector_db import VectorDatabase

    embedder = EmbeddingGenerator(config)
    db = VectorDatabase(config)

    print(f"\nSearching for: '{args.query}'")
    query_emb = embedder.generate_one(args.query)
    results = db.search(query_vector=query_emb, k=5)

    print(f"\nFound {len(results)} results:")
    for i, r in enumerate(results, 1):
        print(f"\n  {i}. Similarity: {r.similarity:.3f}")
        print(f"     Source: {r.metadata.get('source')}")
        print(f"     File: {r.metadata.get('file_path', 'unknown')}")
        print(f"     Preview: {r.text[:150]}...")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()

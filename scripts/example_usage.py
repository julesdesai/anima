#!/usr/bin/env python3
"""
Example usage of the Castor system.

This demonstrates the basic workflow:
1. Setup database
2. Ingest corpus
3. Query with different models
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.vector_db import VectorDatabase
from src.corpus.ingest import CorpusIngester
from src.agent.factory import AgentFactory
from src.config import get_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    """Run example workflow"""
    print("\n" + "="*60)
    print("Castor - Example Usage")
    print("="*60 + "\n")

    # Load configuration
    config = get_config()
    logger.info("Configuration loaded")

    # 1. Setup database
    print("Step 1: Setting up vector database...")
    db = VectorDatabase(config)
    db.create_collection(force=False)
    print("✓ Database ready\n")

    # 2. Check if corpus is ingested
    info = db.get_collection_info()
    if info.get("points_count", 0) == 0:
        print("Step 2: Ingesting sample corpus...")
        ingester = CorpusIngester(config)
        count = ingester.ingest_directory(
            directory=config.user.corpus_path,
            recursive=True,
            force_recreate=False,
        )
        print(f"✓ Ingested {count} documents\n")
    else:
        print(f"Step 2: Corpus already ingested ({info.get('points_count')} documents)\n")

    # 3. Create agent
    print("Step 3: Creating agent...")
    print(f"Using primary model: {config.model.primary}")

    try:
        agent = AgentFactory.create_primary(
            user_name=config.user.name,
            config=config,
        )
        print(f"✓ Agent ready: {agent.__class__.__name__}\n")
    except ValueError as e:
        logger.error(f"Failed to create agent: {e}")
        logger.info("Make sure you have set the appropriate API keys in .env")
        print("\n⚠️  Please configure API keys in .env file")
        return

    # 4. Example queries
    print("Step 4: Running example queries...\n")

    queries = [
        "What are my thoughts on AI alignment?",
        "What's my opinion on test-time compute vs fine-tuning?",
        "How do I usually communicate with my advisor?",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 60)

        try:
            result = agent.respond(query)

            print(f"Response:\n{result['response']}\n")
            print(f"[Tool calls: {len(result['tool_calls'])}, "
                  f"Iterations: {result['iterations']}]")

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print(f"Error: {e}")

        print()

    print("="*60)
    print("Example completed successfully!")
    print("="*60 + "\n")

    # Show usage instructions
    print("To use interactively, run:")
    print("  python scripts/chat.py --model claude\n")


if __name__ == "__main__":
    main()

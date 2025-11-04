#!/usr/bin/env python3
"""Initialize the vector database"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.vector_db import VectorDatabase
from src.config import get_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    """Initialize vector database"""
    logger.info("Initializing vector database...")

    config = get_config()
    db = VectorDatabase(config)

    # Create collection
    db.create_collection(force=False)

    logger.info("Vector database initialized successfully!")
    logger.info(f"Collection: {config.vector_db.collection_name}")
    logger.info(f"Host: {config.vector_db.host}:{config.vector_db.port}")


if __name__ == "__main__":
    main()

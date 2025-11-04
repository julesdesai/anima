#!/usr/bin/env python3
"""Ingest corpus files into vector database"""

import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.corpus.ingest import CorpusIngester
from src.config import get_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    """Ingest corpus"""
    parser = argparse.ArgumentParser(description="Ingest corpus into vector database")
    parser.add_argument(
        "--directory",
        "-d",
        type=str,
        default=None,
        help="Directory containing corpus files (default from config)",
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        default=True,
        help="Search directory recursively (default: True)",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Recreate collection (deletes existing data)",
    )
    parser.add_argument(
        "--no-incremental",
        action="store_true",
        help="Process all files (ignore already ingested files)",
    )

    args = parser.parse_args()

    config = get_config()
    ingester = CorpusIngester(config)

    # Use directory from args or config
    directory = args.directory or config.user.corpus_path

    logger.info(f"Ingesting corpus from: {directory}")
    logger.info(f"Recursive: {args.recursive}")
    logger.info(f"Force recreate: {args.force}")
    logger.info(f"Incremental: {not args.no_incremental}")

    # Ingest
    count = ingester.ingest_directory(
        directory=directory,
        recursive=args.recursive,
        force_recreate=args.force,
        incremental=not args.no_incremental,
    )

    logger.info(f"Successfully ingested {count} documents")


if __name__ == "__main__":
    main()

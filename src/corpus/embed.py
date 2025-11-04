"""Embedding generation for text chunks"""

import os
import logging
from typing import List
from openai import OpenAI

from ..config import get_config

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings for text using OpenAI API"""

    def __init__(self, config=None):
        """Initialize embedding generator"""
        if config is None:
            config = get_config()

        self.config = config
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = config.embedding.model
        self.batch_size = config.embedding.batch_size

        logger.info(f"Initialized embedding generator with model: {self.model}")

    def generate(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        if not texts:
            return []

        all_embeddings = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        logger.info(f"Starting embedding generation for {len(texts)} texts in {total_batches} batches")

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_num = i // self.batch_size + 1

            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} texts)...")

            try:
                logger.debug(f"Calling OpenAI API for batch {batch_num}...")
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                )
                logger.debug(f"Received response from OpenAI for batch {batch_num}")

                # Extract embeddings
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

                logger.info(
                    f"✓ Batch {batch_num}/{total_batches} complete "
                    f"({len(all_embeddings)}/{len(texts)} embeddings generated)"
                )

            except Exception as e:
                logger.error(f"Error generating embeddings for batch {batch_num}: {e}")
                raise

        logger.info(f"✓ Generated all {len(all_embeddings)} embeddings successfully")
        return all_embeddings

    def generate_one(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        embeddings = self.generate([text])
        return embeddings[0] if embeddings else []

"""Tool definitions and implementations"""

import logging
from typing import List, Optional, Dict, Any

from ..database.vector_db import VectorDatabase
from ..database.schema import SearchResult, SearchFilters, SourceType
from ..corpus.embed import EmbeddingGenerator
from ..config import get_config

logger = logging.getLogger(__name__)


class CorpusSearchTool:
    """Search tool for corpus retrieval"""

    def __init__(self, config=None):
        """Initialize search tool"""
        if config is None:
            config = get_config()

        self.config = config
        self.db = VectorDatabase(config)
        self.embedder = EmbeddingGenerator(config)
        self._style_pack_cache = None  # Cache diverse style examples

    def get_style_pack(self) -> List[Dict[str, Any]]:
        """
        Get diverse representative writing samples for style grounding.
        Uses caching to avoid recomputing.

        Returns:
            List of diverse document samples
        """
        if self._style_pack_cache is not None:
            return self._style_pack_cache

        if not self.config.retrieval.style_pack_enabled:
            return []

        size = self.config.retrieval.style_pack_size
        logger.info(f"Building style pack with {size} diverse samples...")

        # Get a random sample by searching for common words
        # This gives us a diverse starting set
        seed_query = "the"  # Very common word
        seed_embedding = self.embedder.generate_one(seed_query)

        # Get more results than we need for diversity selection
        candidates = self.db.search(
            query_vector=seed_embedding,
            k=size * 5,  # Get 5x to select diverse subset
        )

        if not candidates:
            logger.warning("No documents found for style pack")
            return []

        # Simple diversity selection: pick documents from different sources/times
        diverse_samples = []
        seen_sources = set()

        for result in candidates:
            source = result.metadata.get("source", "unknown")
            file_path = result.metadata.get("file_path", "")

            # Prioritize diversity in source and file
            key = f"{source}:{file_path}"

            if key not in seen_sources or len(diverse_samples) < size:
                diverse_samples.append({
                    "text": result.text,
                    "metadata": result.metadata,
                    "similarity": result.similarity,
                })
                seen_sources.add(key)

            if len(diverse_samples) >= size:
                break

        logger.info(f"Style pack created with {len(diverse_samples)} samples from {len(seen_sources)} sources")
        self._style_pack_cache = diverse_samples
        return diverse_samples

    def search(
        self,
        query: str,
        k: Optional[int] = None,
        time_range: Optional[Dict[str, Optional[str]]] = None,
        source_filter: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search the user's corpus for relevant text.

        Args:
            query: Search query
            k: Number of results to return (default from config)
            time_range: Optional time filter with 'start' and 'end' ISO timestamps
            source_filter: Optional list of source types to filter by

        Returns:
            List of search results with text, metadata, and similarity scores
        """
        # Use default k if not provided
        if k is None:
            k = self.config.retrieval.default_k

        # Validate k
        k = min(k, self.config.retrieval.max_k)

        logger.debug(f"Searching corpus for: '{query}' (k={k})")

        # Generate query embedding
        query_embedding = self.embedder.generate_one(query)

        # Build filters
        filters = None
        if time_range or source_filter:
            filters = SearchFilters(
                time_range=time_range,
                source_filter=[SourceType(s) for s in source_filter]
                if source_filter
                else None,
            )

        # Execute hybrid search (combines semantic + keyword matching)
        results = self.db.hybrid_search(
            query_text=query,
            query_vector=query_embedding,
            k=k,
            filters=filters,
        )

        # Note: Hybrid search uses RRF scores (or semantic scores), ranked by relevance
        logger.info(
            f"Hybrid search '{query}' (k={k}): Found {len(results)} results"
        )

        if results:
            logger.info(
                f"  Top result score: {results[0].similarity:.3f}, "
                f"Avg score: {sum(r.similarity for r in results)/len(results):.3f}"
            )

            # Log preview of top 3 results for debugging
            logger.debug("Top 3 results:")
            for i, r in enumerate(results[:3], 1):
                preview = r.text[:100].replace('\n', ' ')
                source = r.metadata.get('source', 'unknown')
                file_name = r.metadata.get('file_path', '').split('/')[-1] if r.metadata.get('file_path') else 'unknown'
                logger.debug(f"  {i}. [{source}/{file_name}] {preview}...")

        # Convert to dict format for tool response
        return [
            {
                "text": result.text,
                "metadata": result.metadata,
                "similarity": result.similarity,
            }
            for result in results
        ]

    def get_tool_definition_claude(self) -> Dict[str, Any]:
        """Get tool definition for Claude API format"""
        return {
            "name": "search_corpus",
            "description": "Search through the user's writing corpus using semantic similarity. Returns relevant excerpts from their emails, documents, thesis, and notes. IMPORTANT: Always use k=50 or higher to get sufficient context. Higher k means more comprehensive understanding.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query - be specific about what you're looking for. Try different phrasings if first search doesn't return enough results.",
                    },
                    "k": {
                        "type": "integer",
                        "description": f"Number of results to return. Recommended: 50-100 for comprehensive answers. Max: {self.config.retrieval.max_k}. Higher is better for matching fine-tuning performance.",
                        "default": self.config.retrieval.default_k,
                    },
                    "time_range": {
                        "type": "object",
                        "description": "Optional time filter",
                        "properties": {
                            "start": {
                                "type": "string",
                                "description": "ISO datetime string or null",
                            },
                            "end": {
                                "type": "string",
                                "description": "ISO datetime string or null",
                            },
                        },
                    },
                    "source_filter": {
                        "type": "array",
                        "description": "Optional filter by source type",
                        "items": {"enum": ["email", "chat", "document", "code", "note"]},
                    },
                },
                "required": ["query"],
            },
        }

    def get_tool_definition_openai(self) -> Dict[str, Any]:
        """Get tool definition for OpenAI/DeepSeek API format"""
        return {
            "type": "function",
            "function": {
                "name": "search_corpus",
                "description": "Search through the user's writing corpus using semantic similarity. Returns relevant excerpts from their emails, documents, thesis, and notes. IMPORTANT: Always use k=50 or higher to get sufficient context for comprehensive, grounded responses.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query - be specific about what you're looking for. Try different phrasings if first search doesn't return enough results.",
                        },
                        "k": {
                            "type": "integer",
                            "description": f"Number of results to return. Recommended: 50-100 for comprehensive answers. Max: {self.config.retrieval.max_k}",
                            "default": self.config.retrieval.default_k,
                        },
                        "time_range": {
                            "type": "object",
                            "description": "Optional time filter",
                            "properties": {
                                "start": {"type": "string"},
                                "end": {"type": "string"},
                            },
                        },
                        "source_filter": {
                            "type": "array",
                            "description": "Optional filter by source type",
                            "items": {
                                "type": "string",
                                "enum": ["email", "chat", "document", "code", "note"],
                            },
                        },
                    },
                    "required": ["query"],
                },
            },
        }

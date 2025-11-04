#!/usr/bin/env python3
"""Add text index to existing collection for hybrid search"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType, TextIndexParams, TokenizerType

def main():
    config = get_config()

    client = QdrantClient(
        host=config.vector_db.host,
        port=config.vector_db.port,
    )

    collection_name = config.vector_db.collection_name

    print(f"Adding text index to collection: {collection_name}")

    try:
        # Try with simplified parameters
        client.create_payload_index(
            collection_name=collection_name,
            field_name="text",
            field_schema=PayloadSchemaType.TEXT,
        )
        print("✓ Text index created successfully!")
        print("\nHybrid search is now enabled.")

    except Exception as e:
        if "already exists" in str(e).lower():
            print("✓ Text index already exists!")
        else:
            print(f"✗ Error creating text index: {e}")
            return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())

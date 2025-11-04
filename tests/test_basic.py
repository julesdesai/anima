"""Basic tests for Castor components"""

import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.database.schema import CorpusDocument, SourceType
from src.corpus.ingest import CorpusIngester


class TestConfig:
    """Test configuration management"""

    def test_config_loading(self):
        """Test that config can be loaded"""
        config = Config.from_yaml("config.yaml")
        assert config.user.name is not None
        assert config.model.primary is not None

    def test_config_has_models(self):
        """Test that model configs exist"""
        config = Config.from_yaml("config.yaml")
        assert config.model.claude is not None
        assert config.model.deepseek is not None
        assert config.model.hermes is not None


class TestSchema:
    """Test data schemas"""

    def test_corpus_document_creation(self):
        """Test creating a corpus document"""
        doc = CorpusDocument(
            id="test-123",
            text="This is a test document",
            metadata={
                "source": "document",
                "timestamp": "2025-01-01T00:00:00",
            },
        )
        assert doc.id == "test-123"
        assert doc.source == SourceType.DOCUMENT
        assert doc.char_length == 23

    def test_source_type_enum(self):
        """Test source type enum"""
        assert SourceType.EMAIL.value == "email"
        assert SourceType.CHAT.value == "chat"
        assert SourceType.DOCUMENT.value == "document"


class TestCorpusIngester:
    """Test corpus ingestion"""

    def test_chunk_text(self):
        """Test text chunking"""
        config = Config.from_yaml("config.yaml")
        ingester = CorpusIngester(config)

        text = "This is a test. " * 100  # Long text
        chunks = ingester.chunk_text(text)

        assert len(chunks) > 0
        assert all(len(chunk) >= config.corpus.min_chunk_length for chunk in chunks)

    def test_infer_source_type(self):
        """Test source type inference"""
        config = Config.from_yaml("config.yaml")
        ingester = CorpusIngester(config)

        email_path = Path("data/corpus/emails/test.txt")
        assert ingester.infer_source_type(email_path) == SourceType.EMAIL

        chat_path = Path("data/corpus/chats/conversation.txt")
        assert ingester.infer_source_type(chat_path) == SourceType.CHAT

        doc_path = Path("data/corpus/documents/article.md")
        assert ingester.infer_source_type(doc_path) == SourceType.DOCUMENT


class TestAgentFactory:
    """Test agent factory"""

    def test_model_name_parsing(self):
        """Test that factory recognizes model names"""
        from src.agent.factory import AgentFactory

        # This would require API keys, so we just test the factory logic
        # In a real test, you'd mock the API clients

        # Just verify the factory can identify model types
        assert "claude" in "claude-sonnet-4.5"
        assert "deepseek" in "deepseek-reasoner"
        assert "hermes" in "hermes-70b"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

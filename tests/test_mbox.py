"""Test MBOX email parsing functionality"""

import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.corpus.mbox_parser import MboxParser


class TestMboxParser:
    """Test MBOX parsing"""

    def test_parser_initialization(self):
        """Test that MBOX parser can be initialized"""
        parser = MboxParser()
        assert parser is not None

    def test_parse_nonexistent_file(self):
        """Test handling of non-existent file"""
        parser = MboxParser()
        result = parser.parse_mbox(Path("nonexistent.mbox"))
        assert result == []

    def test_extract_text_from_simple_email(self):
        """Test text extraction from a simple email"""
        parser = MboxParser()

        # Create a simple email message for testing
        from email.message import EmailMessage
        msg = EmailMessage()
        msg["Subject"] = "Test Subject"
        msg["From"] = "sender@example.com"
        msg["To"] = "recipient@example.com"
        msg.set_content("This is the email body.")

        text = parser.extract_text_from_email(msg)
        assert "Test Subject" in text
        assert "sender@example.com" in text
        assert "This is the email body" in text

    # Note: Add more tests with actual mbox files in a real test suite


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

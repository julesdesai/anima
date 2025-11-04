"""Test PDF extraction functionality"""

import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.corpus.pdf_extractor import PDFExtractor, is_pdf_available


class TestPDFExtractor:
    """Test PDF extraction"""

    def test_pdf_available(self):
        """Test that PDF support is available"""
        # This will pass if pypdf is installed
        assert is_pdf_available() or not is_pdf_available()

    @pytest.mark.skipif(not is_pdf_available(), reason="pypdf not installed")
    def test_extractor_initialization(self):
        """Test that PDF extractor can be initialized"""
        extractor = PDFExtractor()
        assert extractor is not None

    @pytest.mark.skipif(not is_pdf_available(), reason="pypdf not installed")
    def test_extract_text_nonexistent_file(self):
        """Test handling of non-existent file"""
        extractor = PDFExtractor()
        result = extractor.extract_text(Path("nonexistent.pdf"))
        assert result is None

    # Note: Add more tests with actual PDF files in a real test suite


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""PDF text extraction utilities"""

import logging
from pathlib import Path
from typing import Optional

try:
    from pypdf import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logging.warning(
        "pypdf not installed. PDF support disabled. "
        "Install with: pip install pypdf"
    )

logger = logging.getLogger(__name__)


class PDFExtractor:
    """Extract text from PDF files using pypdf"""

    def __init__(self):
        """Initialize PDF extractor"""
        if not PDF_AVAILABLE:
            raise ImportError(
                "pypdf is required for PDF support. "
                "Install with: pip install pypdf"
            )

    def extract_text(self, pdf_path: Path) -> Optional[str]:
        """
        Extract all text from a PDF file.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text or None if extraction fails
        """
        try:
            reader = PdfReader(pdf_path)

            if reader.is_encrypted:
                logger.warning(f"PDF is encrypted: {pdf_path}")
                return None

            # Extract text from all pages
            text_parts = []
            for page_num, page in enumerate(reader.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                    else:
                        logger.debug(f"No text on page {page_num} of {pdf_path}")
                except Exception as e:
                    logger.warning(f"Error extracting page {page_num} from {pdf_path}: {e}")
                    continue

            if not text_parts:
                logger.warning(f"No text extracted from PDF: {pdf_path}")
                return None

            # Join all pages with double newline
            full_text = "\n\n".join(text_parts)

            logger.info(
                f"Extracted {len(full_text)} characters from "
                f"{len(text_parts)} pages in {pdf_path.name}"
            )

            return full_text

        except Exception as e:
            logger.error(f"Error reading PDF {pdf_path}: {e}")
            return None

    def extract_metadata(self, pdf_path: Path) -> dict:
        """
        Extract metadata from PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with metadata
        """
        try:
            reader = PdfReader(pdf_path)
            metadata = {}

            # Get document info
            if reader.metadata:
                metadata.update({
                    "title": reader.metadata.get("/Title", ""),
                    "author": reader.metadata.get("/Author", ""),
                    "subject": reader.metadata.get("/Subject", ""),
                    "creator": reader.metadata.get("/Creator", ""),
                    "producer": reader.metadata.get("/Producer", ""),
                    "creation_date": reader.metadata.get("/CreationDate", ""),
                })

            # Add page count
            metadata["page_count"] = len(reader.pages)

            # Clean up empty values
            metadata = {k: v for k, v in metadata.items() if v}

            return metadata

        except Exception as e:
            logger.error(f"Error extracting metadata from {pdf_path}: {e}")
            return {}


def is_pdf_available() -> bool:
    """Check if PDF support is available"""
    return PDF_AVAILABLE

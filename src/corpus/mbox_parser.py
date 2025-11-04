"""MBOX email archive parser"""

import mailbox
import logging
from pathlib import Path
from typing import List, Optional, Dict
from email.utils import parsedate_to_datetime
from datetime import datetime
import email
from email.parser import BytesParser
from email.policy import default

logger = logging.getLogger(__name__)


class MboxParser:
    """Parse MBOX email archives"""

    def __init__(self):
        """Initialize MBOX parser"""
        pass

    def extract_text_from_email(self, message) -> str:
        """
        Extract text content from an email message.

        Args:
            message: Email message object

        Returns:
            Extracted text content
        """
        text_parts = []

        # Get subject
        subject = message.get("Subject", "")
        if subject:
            text_parts.append(f"Subject: {subject}")

        # Get from/to
        from_addr = message.get("From", "")
        to_addr = message.get("To", "")
        if from_addr:
            text_parts.append(f"From: {from_addr}")
        if to_addr:
            text_parts.append(f"To: {to_addr}")

        # Add separator
        if text_parts:
            text_parts.append("")  # blank line

        # Extract body
        if message.is_multipart():
            # Handle multipart messages
            for part in message.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition", ""))

                # Skip attachments
                if "attachment" in content_disposition:
                    continue

                # Get text parts
                if content_type == "text/plain":
                    try:
                        payload = part.get_payload(decode=True)
                        if payload:
                            text = payload.decode("utf-8", errors="ignore")
                            text_parts.append(text)
                    except Exception as e:
                        logger.debug(f"Error decoding part: {e}")
                        continue

                elif content_type == "text/html":
                    # Try to extract text from HTML (basic)
                    try:
                        payload = part.get_payload(decode=True)
                        if payload:
                            html = payload.decode("utf-8", errors="ignore")
                            # Basic HTML stripping (remove tags)
                            import re
                            text = re.sub(r"<[^>]+>", "", html)
                            text = re.sub(r"\s+", " ", text)
                            if text.strip():
                                text_parts.append(text)
                    except Exception as e:
                        logger.debug(f"Error decoding HTML: {e}")
                        continue

        else:
            # Handle simple messages
            try:
                payload = message.get_payload(decode=True)
                if payload:
                    text = payload.decode("utf-8", errors="ignore")
                    text_parts.append(text)
            except Exception as e:
                logger.debug(f"Error decoding message: {e}")

        return "\n".join(text_parts)

    def get_email_metadata(self, message) -> Dict:
        """
        Extract metadata from email message.

        Args:
            message: Email message object

        Returns:
            Dictionary with metadata
        """
        metadata = {}

        # Get basic fields
        metadata["subject"] = message.get("Subject", "")
        metadata["from"] = message.get("From", "")
        metadata["to"] = message.get("To", "")
        metadata["cc"] = message.get("Cc", "")

        # Get date
        date_str = message.get("Date")
        if date_str:
            try:
                date_obj = parsedate_to_datetime(date_str)
                metadata["date"] = date_obj.isoformat()
            except Exception as e:
                logger.debug(f"Error parsing date: {e}")
                metadata["date"] = date_str

        # Get message ID
        metadata["message_id"] = message.get("Message-ID", "")

        # Clean up empty values
        metadata = {k: v for k, v in metadata.items() if v}

        return metadata

    def parse_mbox(self, mbox_path: Path) -> List[Dict]:
        """
        Parse an MBOX file and extract all emails.

        Args:
            mbox_path: Path to MBOX file

        Returns:
            List of dictionaries with email data
        """
        emails = []

        try:
            # Open mbox file
            mbox = mailbox.mbox(str(mbox_path))

            logger.info(f"Parsing MBOX file: {mbox_path}")

            # Process each message
            for idx, message in enumerate(mbox):
                try:
                    # Extract text content
                    text = self.extract_text_from_email(message)

                    if not text.strip():
                        logger.debug(f"Empty email at index {idx}")
                        continue

                    # Extract metadata
                    metadata = self.get_email_metadata(message)

                    # Add to results
                    emails.append({
                        "text": text,
                        "metadata": metadata,
                        "index": idx,
                    })

                except Exception as e:
                    logger.warning(f"Error processing email {idx}: {e}")
                    continue

            logger.info(f"Extracted {len(emails)} emails from {mbox_path.name}")

        except Exception as e:
            logger.error(f"Error reading MBOX file {mbox_path}: {e}")

        return emails

    def parse_mbox_to_text(self, mbox_path: Path) -> str:
        """
        Parse MBOX file and return all emails as concatenated text.

        Args:
            mbox_path: Path to MBOX file

        Returns:
            All emails concatenated with separators
        """
        emails = self.parse_mbox(mbox_path)

        text_parts = []
        for email_data in emails:
            text_parts.append(email_data["text"])
            text_parts.append("\n" + "="*80 + "\n")  # Separator

        return "\n".join(text_parts)

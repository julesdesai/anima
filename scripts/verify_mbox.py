#!/usr/bin/env python3
"""Verify MBOX file and show preview of extracted emails"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.corpus.mbox_parser import MboxParser


def main():
    """Verify and preview MBOX file"""
    parser = argparse.ArgumentParser(description="Verify MBOX email file")
    parser.add_argument(
        "mbox_file",
        type=str,
        help="Path to MBOX file",
    )
    parser.add_argument(
        "--preview",
        "-p",
        type=int,
        default=3,
        help="Number of emails to preview (default: 3)",
    )

    args = parser.parse_args()

    mbox_path = Path(args.mbox_file)

    if not mbox_path.exists():
        print(f"Error: File not found: {mbox_path}")
        return 1

    print(f"\n{'='*60}")
    print(f"Verifying MBOX file: {mbox_path.name}")
    print(f"{'='*60}\n")

    # Parse mbox
    mbox_parser = MboxParser()
    emails = mbox_parser.parse_mbox(mbox_path)

    if not emails:
        print("❌ No emails found in MBOX file")
        return 1

    print(f"✅ Successfully parsed {len(emails)} emails\n")

    # Show preview
    preview_count = min(args.preview, len(emails))
    print(f"Previewing first {preview_count} emails:\n")

    for i, email_data in enumerate(emails[:preview_count], 1):
        print(f"\n--- Email {i} ---")

        # Show metadata
        metadata = email_data.get("metadata", {})
        if "subject" in metadata:
            print(f"Subject: {metadata['subject']}")
        if "from" in metadata:
            print(f"From: {metadata['from']}")
        if "to" in metadata:
            print(f"To: {metadata['to']}")
        if "date" in metadata:
            print(f"Date: {metadata['date']}")

        # Show text preview (first 200 chars)
        text = email_data.get("text", "")
        text_preview = text[:200] + "..." if len(text) > 200 else text
        print(f"\nContent preview:\n{text_preview}")
        print(f"\n(Total length: {len(text)} characters)")

    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  Total emails: {len(emails)}")
    print(f"  Total characters: {sum(len(e['text']) for e in emails):,}")
    print(f"  Average email length: {sum(len(e['text']) for e in emails) // len(emails):,} chars")
    print(f"{'='*60}\n")

    print("To ingest this MBOX file, run:")
    print(f"  python scripts/ingest_corpus.py\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())

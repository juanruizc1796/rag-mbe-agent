"""
Standalone ingestion script.
Usage:
    python scripts/ingest.py --pdf-dir /path/to/pdfs
or inside Docker:
    docker exec mbe_app python scripts/ingest.py --pdf-dir /app/data/pdfs
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into FAISS index.")
    parser.add_argument(
        "--pdf-dir",
        required=True,
        help="Path to directory containing PDF files.",
    )
    args = parser.parse_args()

    from utils.rag import ingest_pdfs

    logger.info("Starting ingestion from: %s", args.pdf_dir)
    try:
        ingest_pdfs(args.pdf_dir)
        logger.info("Ingestion complete.")
    except FileNotFoundError as exc:
        logger.error("Directory not found: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.error("Ingestion failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

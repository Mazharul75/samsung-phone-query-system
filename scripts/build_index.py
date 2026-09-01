"""Phase 2 entry point: build the RAG vector store from the database.

Run as a module. See the README for usage.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter

from src.database.connection import session_scope
from src.database.repository import PhoneRepository
from src.rag.documents import build_corpus
from src.rag.vector_store import HybridVectorStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the FAISS + BM25 index used by the chatbot."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("build_index")

    with session_scope() as session:
        repository = PhoneRepository(session)
        if repository.count_phones() == 0:
            logger.error(
                "The database is empty. Run 'python -m scripts.run_scraper' "
                "or 'python -m scripts.load_dataset' first."
            )
            return 1
        documents = build_corpus(repository)

    store = HybridVectorStore()
    store.build(documents)
    location = store.save()

    sections = Counter(document.section for document in documents)
    phones = {document.phone_name for document in documents}

    print("\n" + "=" * 62)
    print("VECTOR STORE BUILT")
    print("=" * 62)
    print(f"Documents : {len(documents)}")
    print(f"Phones    : {len(phones)}")
    print(f"Location  : {location}")
    print("\nDocuments per section:")
    for section, count in sections.most_common():
        print(f"  {section:<18} {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

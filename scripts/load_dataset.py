"""Rebuild the database from the dataset shipped with the repository.

This is the offline alternative to running the scraper: it needs no network
access and completes in a couple of seconds. See the README for usage.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from src.scraper.pipeline import load_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Load data/samsung_phones_dataset.json into MySQL."
    )
    parser.add_argument(
        "--path",
        default=None,
        help="Path to a dataset JSON file (defaults to the shipped dataset).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate all tables before loading (schema changes).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        summary = load_dataset(args.path, recreate=args.reset)
    except FileNotFoundError:
        logging.error("Dataset file not found. Run 'python -m scripts.run_scraper'.")
        return 1
    except Exception as exc:
        logging.error("Load failed: %s", exc)
        return 1

    print("\n" + "=" * 62)
    print("DATASET LOAD SUMMARY")
    print("=" * 62)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

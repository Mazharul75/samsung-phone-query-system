"""Phase 1 entry point: scrape GSMArena and populate the database.

Run as a module. See the README for the available options, or pass ``--help``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from src.config import settings
from src.scraper.pipeline import run_scrape
from src.scraper.targets import TARGET_MODELS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scrape Samsung phone specifications from GSMArena into MySQL."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        metavar="NAME",
        help="Override the built-in model list (GSMArena listing names).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore cached HTML and re-download every page.",
    )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Scrape and snapshot only; do not write to the database.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Seconds to wait between requests (default: %(default)s).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate all tables before loading (schema changes).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.no_cache:
        settings.scraper_use_cache = False
    if args.delay is not None:
        settings.scraper_delay_seconds = args.delay

    targets = tuple(args.models) if args.models else TARGET_MODELS

    try:
        summary = run_scrape(
            targets, persist=not args.no_persist, recreate=args.reset
        )
    except Exception as exc:
        logging.getLogger("scraper").error("Scrape failed: %s", exc)
        return 1

    print("\n" + "=" * 62)
    print("SCRAPE SUMMARY")
    print("=" * 62)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

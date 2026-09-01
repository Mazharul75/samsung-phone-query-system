"""Phase 3 entry point: run the multi-agent crew to produce a review.

Takes a phone model as its argument. See the README for usage, or pass
``--help``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.agents.crew import get_crew


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a product review with the multi-agent crew."
    )
    parser.add_argument("phone", help="Phone model, e.g. 'Galaxy S24 Ultra'.")
    parser.add_argument(
        "--save", metavar="PATH", help="Write the review markdown to a file."
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

    result = get_crew().run(args.phone)

    print("\n" + "=" * 70)
    print("AGENT TRANSCRIPT")
    print("=" * 70)
    for step in result.transcript:
        status = "OK " if step.success else "FAIL"
        print(f"[{status}] {step.agent:<22} {step.duration_seconds:6.2f}s  "
              f"{step.summary or step.error}")

    if not result.success:
        print(f"\nReview generation failed: {result.error}")
        return 1

    print("\n" + "=" * 70)
    print("GENERATED REVIEW")
    print("=" * 70)
    print(result.review["markdown"])
    print(f"\nTotal time: {result.duration_seconds:.1f}s")

    if args.save:
        path = Path(args.save)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result.review["markdown"], encoding="utf-8")
        print(f"Saved to {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

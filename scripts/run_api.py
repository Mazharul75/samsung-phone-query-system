"""Phase 4 entry point: start the REST API server.

Run as a module. See the README for usage, or pass ``--help``.
"""

from __future__ import annotations

import argparse
import sys

import uvicorn

from src.config import settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the FastAPI server.")
    parser.add_argument("--host", default=settings.api_host)
    parser.add_argument("--port", type=int, default=settings.api_port)
    parser.add_argument(
        "--reload", action="store_true", help="Reload on code changes (development)."
    )
    args = parser.parse_args(argv)

    print(f"Interactive API documentation: http://127.0.0.1:{args.port}/docs")
    uvicorn.run(
        "src.api.main:app", host=args.host, port=args.port, reload=args.reload
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

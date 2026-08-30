"""A polite, cached HTTP client for scraping GSMArena.

Three concerns are handled here so the scraping logic stays readable:

* **Politeness** - a fixed delay is enforced between requests and a real
  browser ``User-Agent`` is sent, matching what ``robots.txt`` allows.
* **Resilience** - transient failures (timeouts, 5xx, 429) are retried with
  exponential backoff instead of aborting a long crawl.
* **Caching** - every downloaded page is written to ``data/raw``. Re-running
  the scraper to fix a parsing bug then costs zero network requests, which is
  both faster and considerably kinder to the source site.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

import requests
from requests import Response, Session
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import RAW_DATA_DIR, settings

logger = logging.getLogger(__name__)


class ScrapeError(RuntimeError):
    """Raised when a page cannot be retrieved after all retries."""


class PoliteHTTPClient:
    """Rate-limited HTTP client with retry and on-disk response caching."""

    def __init__(
        self,
        *,
        delay_seconds: float | None = None,
        timeout: int | None = None,
        use_cache: bool | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self.delay_seconds = (
            settings.scraper_delay_seconds if delay_seconds is None else delay_seconds
        )
        self.timeout = settings.scraper_timeout_seconds if timeout is None else timeout
        self.use_cache = settings.scraper_use_cache if use_cache is None else use_cache
        self.cache_dir = cache_dir or (RAW_DATA_DIR / "html")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._last_request_at = 0.0
        self._session: Session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": settings.scraper_user_agent,
                "Accept": (
                    "text/html,application/xhtml+xml,application/xml;q=0.9,"
                    "image/avif,image/webp,*/*;q=0.8"
                ),
                "Accept-Language": "en-US,en;q=0.9",
                "Connection": "keep-alive",
            }
        )

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------
    def _cache_path(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        tail = url.rstrip("/").rsplit("/", 1)[-1].replace(".php", "") or "index"
        safe_tail = "".join(c for c in tail if c.isalnum() or c in "-_")[:60]
        return self.cache_dir / f"{safe_tail}-{digest}.html"

    def _read_cache(self, url: str) -> str | None:
        path = self._cache_path(url)
        if self.use_cache and path.exists():
            logger.debug("Cache hit for %s", url)
            return path.read_text(encoding="utf-8")
        return None

    def _write_cache(self, url: str, html: str) -> None:
        if self.use_cache:
            self._cache_path(url).write_text(html, encoding="utf-8")

    # ------------------------------------------------------------------
    # Fetching
    # ------------------------------------------------------------------
    def _throttle(self) -> None:
        """Sleep just long enough to honour the configured request delay."""
        elapsed = time.monotonic() - self._last_request_at
        remaining = self.delay_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_request_at = time.monotonic()

    @retry(
        retry=retry_if_exception_type((requests.RequestException, ScrapeError)),
        stop=stop_after_attempt(settings.scraper_max_retries),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        reraise=True,
    )
    def _request(self, url: str) -> Response:
        self._throttle()
        logger.debug("GET %s", url)
        response = self._session.get(url, timeout=self.timeout)
        if response.status_code == 429:
            raise ScrapeError(f"Rate limited (429) on {url}")
        if response.status_code >= 500:
            raise ScrapeError(f"Server error {response.status_code} on {url}")
        response.raise_for_status()
        return response

    def get_html(self, url: str) -> str:
        """Return the HTML for ``url``, using the disk cache when available."""
        cached = self._read_cache(url)
        if cached is not None:
            return cached

        try:
            response = self._request(url)
        except requests.RequestException as exc:
            raise ScrapeError(f"Failed to fetch {url}: {exc}") from exc

        # GSMArena serves windows-1252 for some legacy pages; letting requests
        # guess from the declared charset keeps accented characters intact.
        response.encoding = response.apparent_encoding or response.encoding
        html = response.text
        self._write_cache(url, html)
        return html

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "PoliteHTTPClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

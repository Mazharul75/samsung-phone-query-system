"""GSMArena scraper for Samsung phone specifications.

The scraper runs in two stages:

1. **Discovery** - walk the paginated Samsung brand listing
   (``samsung-phones-9.php``) and build a ``model name -> detail URL`` map.
   Resolving URLs from the live listing rather than hard-coding them means the
   scraper keeps working when GSMArena renumbers a page.
2. **Extraction** - request each detail page and parse the specification
   tables into a :class:`ScrapedPhone`, capturing *every* published row plus a
   set of normalised numeric fields.

Only pages that ``robots.txt`` permits are requested, and all traffic goes
through :class:`~src.scraper.http_client.PoliteHTTPClient`, which rate-limits
and caches.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Tag

from src.config import settings
from src.scraper.http_client import PoliteHTTPClient, ScrapeError
from src.scraper.parsers import (
    clean_text,
    parse_battery_capacity,
    parse_camera_megapixels,
    parse_charging_watts,
    parse_display_size,
    parse_endurance,
    parse_endurance_hours,
    parse_memory_options,
    parse_prices,
    parse_refresh_rate,
    parse_release_year,
    parse_weight,
    slugify,
)
from src.scraper.targets import TARGET_LOOKUP, TARGET_MODELS, normalise_model_name

logger = logging.getLogger(__name__)


@dataclass
class ScrapedPhone:
    """Everything extracted from one GSMArena detail page.

    ``specs`` keeps the complete, lossless representation of the page while the
    scalar attributes hold the normalised values used for SQL comparisons.
    """

    name: str
    slug: str
    source_url: str
    image_url: str | None = None

    #: ``[(category, key, value, position), ...]`` in original page order.
    specs: list[tuple[str, str, str, int]] = field(default_factory=list)
    #: ``[(currency, amount, raw_text), ...]``
    prices: list[tuple[str, float, str]] = field(default_factory=list)

    announced: str | None = None
    release_status: str | None = None
    release_year: int | None = None

    dimensions: str | None = None
    weight_grams: float | None = None
    build: str | None = None

    display_type: str | None = None
    display_size_inches: float | None = None
    display_resolution: str | None = None
    refresh_rate_hz: int | None = None
    display_protection: str | None = None

    operating_system: str | None = None
    chipset: str | None = None
    cpu: str | None = None
    gpu: str | None = None

    memory_internal: str | None = None
    max_ram_gb: int | None = None
    max_storage_gb: int | None = None
    card_slot: str | None = None

    main_camera_summary: str | None = None
    main_camera_mp: float | None = None
    main_camera_video: str | None = None
    selfie_camera_summary: str | None = None
    selfie_camera_mp: float | None = None
    selfie_camera_video: str | None = None

    battery_type: str | None = None
    battery_capacity_mah: int | None = None
    charging: str | None = None
    charging_watts: int | None = None
    battery_endurance: str | None = None
    battery_endurance_hours: float | None = None
    battery_endurance_metric: str | None = None

    colors: str | None = None
    network_technology: str | None = None
    sim: str | None = None

    @property
    def spec_count(self) -> int:
        return len(self.specs)


class GSMArenaScraper:
    """Scrapes Samsung phone specifications from GSMArena."""

    #: First page of the Samsung brand listing.
    BRAND_LISTING = "samsung-phones-9.php"
    #: Template for subsequent listing pages.
    BRAND_LISTING_PAGED = "samsung-phones-f-9-0-p{page}.php"

    def __init__(self, client: PoliteHTTPClient | None = None) -> None:
        self.client = client or PoliteHTTPClient()
        self.base_url = settings.scraper_base_url.rstrip("/") + "/"

    # ------------------------------------------------------------------
    # Stage 1 - discovery
    # ------------------------------------------------------------------
    def _listing_url(self, page: int) -> str:
        path = (
            self.BRAND_LISTING
            if page == 1
            else self.BRAND_LISTING_PAGED.format(page=page)
        )
        return urljoin(self.base_url, path)

    def _parse_listing_page(self, html: str) -> dict[str, str]:
        """Return ``{model name: absolute detail URL}`` for one listing page."""
        soup = BeautifulSoup(html, "lxml")
        found: dict[str, str] = {}
        for anchor in soup.select("div.makers ul li a"):
            label = anchor.select_one("span")
            href = anchor.get("href")
            if not label or not href:
                continue
            name = clean_text(label.get_text(" ", strip=True))
            if name:
                found[name] = urljoin(self.base_url, str(href))
        return found

    def discover_phone_urls(
        self, targets: tuple[str, ...] = TARGET_MODELS
    ) -> dict[str, str]:
        """Resolve each target model name to its GSMArena detail URL.

        Listing pages are walked in order and the crawl stops as soon as every
        target has been found, so a full 29-page crawl is never needed.
        """
        wanted = {normalise_model_name(t) for t in targets}
        resolved: dict[str, str] = {}

        for page in range(1, settings.scraper_max_listing_pages + 1):
            if not wanted:
                break

            url = self._listing_url(page)
            logger.info("Discovering models on listing page %d", page)
            try:
                html = self.client.get_html(url)
            except ScrapeError as exc:
                logger.warning("Listing page %d unavailable: %s", page, exc)
                continue

            for name, detail_url in self._parse_listing_page(html).items():
                key = normalise_model_name(name)
                if key in wanted:
                    canonical = TARGET_LOOKUP.get(key, name)
                    resolved[canonical] = detail_url
                    wanted.discard(key)
                    logger.info("  resolved %-24s -> %s", canonical, detail_url)

        if wanted:
            missing = sorted(TARGET_LOOKUP.get(k, k) for k in wanted)
            logger.warning("Could not resolve %d model(s): %s", len(missing), missing)

        return resolved

    # ------------------------------------------------------------------
    # Stage 2 - extraction
    # ------------------------------------------------------------------
    @staticmethod
    def _cell_text(cell: Tag) -> str | None:
        """Extract a table cell's text, preserving its line breaks.

        GSMArena separates the individual camera lenses inside a single cell
        with ``<br>`` tags::

            50 MP, f/1.8, 24mm (wide), OIS<br>
            10 MP, f/2.4, 70mm (telephoto), 3x optical zoom<br>
            12 MP, f/2.2, 13mm (ultrawide)

        Collapsing that to plain text merges three lenses into one run-on
        string, and every downstream consumer - the database, the retriever and
        the language model - then treats them as a single sensor.  Converting
        the breaks to an explicit separator first keeps the items distinct.
        """
        for line_break in cell.find_all("br"):
            line_break.replace_with(" | ")
        return clean_text(cell.get_text(" ", strip=True))

    @staticmethod
    def _extract_spec_rows(soup: BeautifulSoup) -> list[tuple[str, str, str, int]]:
        """Flatten the detail page's specification tables.

        GSMArena leaves the label cell empty when a row continues the previous
        specification (e.g. a second block of ``4G bands``).  The last seen
        label is carried forward so those rows stay attached to the right key,
        and ``position`` preserves page order while keeping repeated keys
        unique.
        """
        rows: list[tuple[str, str, str, int]] = []

        for table in soup.select("#specs-list table"):
            header = table.select_one("th")
            if not header:
                continue
            category = clean_text(header.get_text(" ", strip=True))
            if not category:
                continue

            last_key = ""
            position = 0
            for row in table.select("tr"):
                value_cell = row.select_one("td.nfo")
                if not value_cell:
                    continue

                label_cell = row.select_one("td.ttl")
                label = (
                    clean_text(label_cell.get_text(" ", strip=True))
                    if label_cell
                    else None
                )
                if label:
                    last_key = label

                value = GSMArenaScraper._cell_text(value_cell)
                if not value or value == "-":
                    continue

                rows.append((category, last_key or category, value, position))
                position += 1

        return rows

    @staticmethod
    def _lookup(
        specs: list[tuple[str, str, str, int]], category: str, key: str
    ) -> str | None:
        """Return the first value matching ``category``/``key``.

        Matching is case-insensitive and ignores the continuation rows, which
        is what callers want for headline fields.
        """
        category_l, key_l = category.casefold(), key.casefold()
        for spec_category, spec_key, value, _position in specs:
            if spec_category.casefold() == category_l and spec_key.casefold() == key_l:
                return value
        return None

    @staticmethod
    def _lookup_category_prefix(
        specs: list[tuple[str, str, str, int]], prefix: str, key: str
    ) -> str | None:
        """Like :meth:`_lookup` but matches a category by prefix.

        GSMArena labels the camera tables ``Main Camera`` / ``Selfie camera``
        but occasionally appends a qualifier, so an exact match is too strict.
        """
        prefix_l, key_l = prefix.casefold(), key.casefold()
        for spec_category, spec_key, value, _position in specs:
            if spec_category.casefold().startswith(prefix_l) and (
                spec_key.casefold() == key_l
            ):
                return value
        return None

    @staticmethod
    def _lookup_battery_test(specs: list[tuple[str, str, str, int]]) -> str | None:
        """Find GSMArena's battery test result under any of its labels.

        The test table is titled ``Our Tests`` (sometimes just ``Tests``).
        Recent phones report the current metric under ``Battery``, while models
        tested before the methodology changed only carry ``Battery (old)``.
        The modern figure is preferred when both are present.
        """
        candidates: list[tuple[str, str]] = [
            (spec_key, value)
            for spec_category, spec_key, value, _position in specs
            if "tests" in spec_category.casefold()
            and spec_key.casefold().startswith("battery")
        ]
        if not candidates:
            return None

        for spec_key, value in candidates:
            if spec_key.casefold() == "battery":
                return value
        return candidates[0][1]

    @classmethod
    def _camera_summary(
        cls, specs: list[tuple[str, str, str, int]], prefix: str
    ) -> str | None:
        """Join every lens row of a camera table into one description."""
        prefix_l = prefix.casefold()
        parts = [
            value
            for spec_category, spec_key, value, _position in specs
            if spec_category.casefold().startswith(prefix_l)
            # "Video" and "Features" describe the camera system as a whole;
            # only the lens rows belong in the lens summary.
            and spec_key.casefold() not in {"video", "features"}
        ]
        return " | ".join(parts) if parts else None

    def parse_phone_page(self, html: str, url: str) -> ScrapedPhone:
        """Parse a GSMArena detail page into a :class:`ScrapedPhone`."""
        soup = BeautifulSoup(html, "lxml")

        title_node = soup.select_one(".specs-phone-name-title")
        if title_node is None:
            raise ScrapeError(f"No phone title found on {url}")
        name = clean_text(title_node.get_text(" ", strip=True))
        if not name:
            raise ScrapeError(f"Empty phone title on {url}")

        image_node = soup.select_one(".specs-photo-main img")
        image_url = None
        if isinstance(image_node, Tag):
            src = image_node.get("src")
            image_url = str(src) if src else None

        specs = self._extract_spec_rows(soup)
        if not specs:
            raise ScrapeError(f"No specification tables found on {url}")

        phone = ScrapedPhone(
            name=name,
            slug=slugify(name),
            source_url=url,
            image_url=image_url,
            specs=specs,
        )

        # -- Launch ----------------------------------------------------
        phone.announced = self._lookup(specs, "Launch", "Announced")
        phone.release_status = self._lookup(specs, "Launch", "Status")
        phone.release_year = parse_release_year(phone.announced, phone.release_status)

        # -- Body ------------------------------------------------------
        phone.dimensions = self._lookup(specs, "Body", "Dimensions")
        weight_raw = self._lookup(specs, "Body", "Weight")
        phone.weight_grams = parse_weight(weight_raw)
        phone.build = self._lookup(specs, "Body", "Build")
        phone.sim = self._lookup(specs, "Body", "SIM")

        # -- Display ---------------------------------------------------
        phone.display_type = self._lookup(specs, "Display", "Type")
        display_size_raw = self._lookup(specs, "Display", "Size")
        phone.display_size_inches = parse_display_size(display_size_raw)
        phone.display_resolution = self._lookup(specs, "Display", "Resolution")
        phone.refresh_rate_hz = parse_refresh_rate(phone.display_type)
        phone.display_protection = self._lookup(specs, "Display", "Protection")

        # -- Platform --------------------------------------------------
        phone.operating_system = self._lookup(specs, "Platform", "OS")
        phone.chipset = self._lookup(specs, "Platform", "Chipset")
        phone.cpu = self._lookup(specs, "Platform", "CPU")
        phone.gpu = self._lookup(specs, "Platform", "GPU")

        # -- Memory ----------------------------------------------------
        phone.memory_internal = self._lookup(specs, "Memory", "Internal")
        phone.max_ram_gb, phone.max_storage_gb = parse_memory_options(
            phone.memory_internal
        )
        phone.card_slot = self._lookup(specs, "Memory", "Card slot")

        # -- Cameras ---------------------------------------------------
        phone.main_camera_summary = self._camera_summary(specs, "Main Camera")
        phone.main_camera_mp = parse_camera_megapixels(phone.main_camera_summary)
        phone.main_camera_video = self._lookup_category_prefix(
            specs, "Main Camera", "Video"
        )
        phone.selfie_camera_summary = self._camera_summary(specs, "Selfie camera")
        phone.selfie_camera_mp = parse_camera_megapixels(phone.selfie_camera_summary)
        phone.selfie_camera_video = self._lookup_category_prefix(
            specs, "Selfie camera", "Video"
        )

        # -- Battery ---------------------------------------------------
        phone.battery_type = self._lookup(specs, "Battery", "Type")
        phone.battery_capacity_mah = parse_battery_capacity(phone.battery_type)
        phone.charging = self._lookup(specs, "Battery", "Charging")
        phone.charging_watts = parse_charging_watts(phone.charging)
        battery_test = self._lookup_battery_test(specs)
        phone.battery_endurance = parse_endurance(battery_test)
        (
            phone.battery_endurance_hours,
            phone.battery_endurance_metric,
        ) = parse_endurance_hours(battery_test)

        # -- Misc ------------------------------------------------------
        phone.colors = self._lookup(specs, "Misc", "Colors")
        phone.network_technology = self._lookup(specs, "Network", "Technology")

        price_raw = self._lookup(specs, "Misc", "Price")
        phone.prices = parse_prices(price_raw)

        return phone

    def scrape_phone(self, url: str) -> ScrapedPhone:
        """Download and parse a single phone detail page."""
        logger.info("Scraping %s", url)
        html = self.client.get_html(url)
        phone = self.parse_phone_page(html, url)
        logger.info(
            "  %-24s %3d specs, %d price(s)",
            phone.name,
            phone.spec_count,
            len(phone.prices),
        )
        return phone

    def scrape_all(
        self, targets: tuple[str, ...] = TARGET_MODELS
    ) -> list[ScrapedPhone]:
        """Discover and scrape every target model.

        A failure on one phone is logged and skipped so a single broken page
        never aborts the whole run.
        """
        urls = self.discover_phone_urls(targets)
        logger.info("Resolved %d/%d target models.", len(urls), len(targets))

        phones: list[ScrapedPhone] = []
        for name, url in urls.items():
            try:
                phones.append(self.scrape_phone(url))
            except ScrapeError as exc:
                logger.error("Skipping %s: %s", name, exc)
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Unexpected error scraping %s: %s", name, exc)

        return phones

    def close(self) -> None:
        self.client.close()

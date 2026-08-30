"""End-to-end scraping pipeline: GSMArena -> normalised rows -> MySQL."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone

from src.config import DATA_DIR, RAW_DATA_DIR
from src.database.connection import init_database, session_scope
from src.database.repository import PhoneRepository
from src.scraper.gsmarena import GSMArenaScraper, ScrapedPhone
from src.scraper.targets import TARGET_MODELS

logger = logging.getLogger(__name__)

#: Stable, version-controlled copy of the scraped data.  Shipping this file
#: means the project can be set up and demonstrated without re-scraping.
DATASET_PATH = DATA_DIR / "samsung_phones_dataset.json"


def save_snapshot(phones: list[ScrapedPhone]) -> str:
    """Persist the scrape result as JSON.

    Two files are written:

    * a timestamped snapshot under ``data/raw`` that keeps the scrape history
      auditable, and
    * ``data/samsung_phones_dataset.json``, the canonical dataset that ships
      with the repository so the database can be rebuilt offline.
    """
    payload = [asdict(phone) for phone in phones]
    serialised = json.dumps(payload, indent=2, ensure_ascii=False, default=str)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    snapshot_path = RAW_DATA_DIR / f"phones-{timestamp}.json"
    snapshot_path.write_text(serialised, encoding="utf-8")

    DATASET_PATH.write_text(serialised, encoding="utf-8")

    logger.info("Snapshot written to %s", snapshot_path)
    logger.info("Dataset written to %s", DATASET_PATH)
    return str(snapshot_path)


def load_dataset(
    path: str | None = None, *, recreate: bool = False
) -> dict[str, object]:
    """Rebuild the database from the shipped JSON dataset.

    This is the offline counterpart to :func:`run_scrape`: it needs no network
    access, which makes setting the project up fast and reproducible.
    """
    dataset_path = DATA_DIR / "samsung_phones_dataset.json" if path is None else path
    records = json.loads(open(dataset_path, encoding="utf-8").read())

    init_database(recreate=recreate)

    phones = [
        ScrapedPhone(
            **{
                key: (
                    [tuple(item) for item in value]
                    if key in {"specs", "prices"}
                    else value
                )
                for key, value in record.items()
            }
        )
        for record in records
    ]

    with session_scope() as session:
        repository = PhoneRepository(session)
        for phone in phones:
            repository.upsert_phone(phone)

    with session_scope() as session:
        stats = PhoneRepository(session).statistics()

    logger.info("Loaded %d phones from %s", len(phones), dataset_path)
    return {"loaded": len(phones), "source": str(dataset_path), "database": stats}


def run_scrape(
    targets: tuple[str, ...] = TARGET_MODELS,
    *,
    persist: bool = True,
    snapshot: bool = True,
    recreate: bool = False,
) -> dict[str, object]:
    """Scrape every target model and store the result.

    Returns a summary dictionary describing what was collected, which the CLI
    prints and the tests assert against.
    """
    init_database(recreate=recreate)

    scraper = GSMArenaScraper()
    try:
        phones = scraper.scrape_all(targets)
    finally:
        scraper.close()

    if not phones:
        raise RuntimeError(
            "No phones were scraped. Check network connectivity and whether "
            "GSMArena changed its page structure."
        )

    snapshot_path = save_snapshot(phones) if snapshot else None

    stored = 0
    if persist:
        with session_scope() as session:
            repository = PhoneRepository(session)
            for phone in phones:
                repository.upsert_phone(phone)
                stored += 1

    with session_scope() as session:
        stats = PhoneRepository(session).statistics()

    summary: dict[str, object] = {
        "requested": len(targets),
        "scraped": len(phones),
        "stored": stored,
        "total_specifications": sum(p.spec_count for p in phones),
        "snapshot": snapshot_path,
        "database": stats,
    }
    logger.info(
        "Scrape complete: %d/%d models, %d specification rows.",
        len(phones),
        len(targets),
        summary["total_specifications"],
    )
    return summary

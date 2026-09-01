"""Shared pytest fixtures.

Parsing tests run entirely against committed HTML fixtures so they never touch
the network.  Tests that genuinely need MySQL are skipped automatically when no
server is reachable, which keeps the suite runnable on a fresh checkout.
"""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def galaxy_s23_html() -> str:
    """A real GSMArena detail page, captured for offline testing."""
    return (FIXTURE_DIR / "gsmarena_galaxy_s23.html").read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def samsung_listing_html() -> str:
    """The first page of GSMArena's Samsung brand listing."""
    return (FIXTURE_DIR / "gsmarena_samsung_listing.html").read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def database_available() -> bool:
    """Whether a MySQL server with the application schema is reachable."""
    try:
        from src.database.connection import healthcheck

        return healthcheck()
    except Exception:
        return False


@pytest.fixture()
def db_session(database_available):
    """A transactional session, rolled back after the test."""
    if not database_available:
        pytest.skip("MySQL server is not reachable")

    from src.database.connection import get_session_factory

    session = get_session_factory()()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture()
def repository(db_session):
    from src.database.repository import PhoneRepository

    return PhoneRepository(db_session)


@pytest.fixture()
def populated_repository(repository):
    """Repository guaranteed to contain scraped phones."""
    if repository.count_phones() == 0:
        pytest.skip("Database is empty - run 'python -m scripts.run_scraper' first")
    return repository

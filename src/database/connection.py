"""Database engine, session factory and schema bootstrapping."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.config import settings
from src.database.models import Base, SpecCategory

logger = logging.getLogger(__name__)

#: Canonical specification groups, pre-seeded so that ordering in the API and
#: in generated documents is stable regardless of scrape order.
DEFAULT_SPEC_CATEGORIES: tuple[str, ...] = (
    "Network",
    "Launch",
    "Body",
    "Display",
    "Platform",
    "Memory",
    "Main Camera",
    "Selfie camera",
    "Sound",
    "Comms",
    "Features",
    "Battery",
    "Misc",
    "Tests",
)

_engine: Engine | None = None
_SessionFactory: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    """Return (and lazily build) the shared SQLAlchemy engine."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            settings.database_url,
            echo=settings.db_echo,
            pool_pre_ping=True,
            pool_recycle=3600,
            future=True,
        )
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    """Return (and lazily build) the shared session factory."""
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(
            bind=get_engine(), autoflush=False, expire_on_commit=False
        )
    return _SessionFactory


@contextmanager
def session_scope() -> Iterator[Session]:
    """Provide a transactional scope around a series of operations.

    Commits on success, rolls back on any exception, and always closes the
    session.  This is the only place transaction handling is written, so no
    caller can leak a connection.
    """
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_database_if_missing() -> None:
    """Create the application database if the server does not have it yet.

    Connects to the server without selecting a database, which is why a
    separate short-lived engine is used here.
    """
    server_engine = create_engine(settings.server_url, future=True, isolation_level="AUTOCOMMIT")
    try:
        with server_engine.connect() as conn:
            conn.execute(
                text(
                    f"CREATE DATABASE IF NOT EXISTS `{settings.db_name}` "
                    "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                )
            )
        logger.info("Database %r is present.", settings.db_name)
    finally:
        server_engine.dispose()


def seed_spec_categories() -> None:
    """Insert the canonical specification categories once."""
    with session_scope() as session:
        existing = {name for (name,) in session.query(SpecCategory.name).all()}
        for order, name in enumerate(DEFAULT_SPEC_CATEGORIES):
            if name not in existing:
                session.add(SpecCategory(name=name, display_order=order))


def init_database(recreate: bool = False) -> None:
    """Create the database, all tables and the seed rows.

    Safe to run repeatedly - every step is idempotent.

    ``create_all`` only creates *missing* tables; it never alters an existing
    one.  So when the model definitions gain a column, pass ``recreate=True``
    to rebuild the schema from scratch.  That is a safe operation here because
    the entire contents can be restored from the shipped dataset or by
    re-running the scraper.
    """
    create_database_if_missing()
    engine = get_engine()

    if recreate:
        logger.warning("Dropping and recreating all tables.")
        Base.metadata.drop_all(bind=engine)

    Base.metadata.create_all(bind=engine)
    seed_spec_categories()
    logger.info("Schema ready (%d tables).", len(Base.metadata.tables))


def healthcheck() -> bool:
    """Return ``True`` when a trivial query succeeds against the database."""
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:  # pragma: no cover - depends on server state
        logger.warning("Database healthcheck failed: %s", exc)
        return False

"""Data-access layer for the Samsung phone database.

Everything that reads or writes phone data goes through this module, so the
scraper, the RAG pipeline, the agents and the API all share one consistent set
of queries instead of each writing their own SQL.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from src.database.models import Phone, Price, SpecCategory, Specification

logger = logging.getLogger(__name__)

#: Columns copied straight from a ScrapedPhone onto a Phone row.
_SCALAR_FIELDS: tuple[str, ...] = (
    "name",
    "slug",
    "source_url",
    "image_url",
    "announced",
    "release_status",
    "release_year",
    "dimensions",
    "weight_grams",
    "build",
    "display_type",
    "display_size_inches",
    "display_resolution",
    "refresh_rate_hz",
    "display_protection",
    "operating_system",
    "chipset",
    "cpu",
    "gpu",
    "memory_internal",
    "max_ram_gb",
    "max_storage_gb",
    "card_slot",
    "main_camera_summary",
    "main_camera_mp",
    "main_camera_video",
    "selfie_camera_summary",
    "selfie_camera_mp",
    "selfie_camera_video",
    "battery_type",
    "battery_capacity_mah",
    "charging",
    "charging_watts",
    "battery_endurance",
    "battery_endurance_hours",
    "battery_endurance_metric",
    "colors",
    "network_technology",
    "sim",
)


def _newest_first():
    """Ordering that lists the newest phones first, with unknown years last.

    MySQL and MariaDB do not implement the SQL standard ``NULLS LAST`` clause,
    so the null check is expressed as a leading boolean sort key instead - a
    portable idiom that also works on PostgreSQL and SQLite.
    """
    return (Phone.release_year.is_(None), Phone.release_year.desc(), Phone.name)


class PhoneRepository:
    """Query and persistence helpers bound to one SQLAlchemy session."""

    def __init__(self, session: Session) -> None:
        self.session = session

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def _category_id(self, name: str) -> int:
        """Return the id of a spec category, creating it when it is new."""
        category = self.session.scalar(
            select(SpecCategory).where(SpecCategory.name == name)
        )
        if category is None:
            max_order = self.session.scalar(
                select(func.coalesce(func.max(SpecCategory.display_order), 0))
            )
            category = SpecCategory(name=name, display_order=(max_order or 0) + 1)
            self.session.add(category)
            self.session.flush()
        return category.id

    def upsert_phone(self, scraped: Any) -> Phone:
        """Insert or update one scraped phone together with its child rows.

        Re-running the scraper must not create duplicates, so an existing row
        is matched on ``slug`` and updated in place.  Specifications and prices
        are replaced wholesale, which is simpler and safer than diffing when
        the source page has been restructured upstream.
        """
        phone = self.session.scalar(select(Phone).where(Phone.slug == scraped.slug))
        if phone is None:
            phone = Phone(slug=scraped.slug, name=scraped.name,
                          source_url=scraped.source_url)
            self.session.add(phone)

        for field_name in _SCALAR_FIELDS:
            setattr(phone, field_name, getattr(scraped, field_name, None))
        phone.brand = "Samsung"
        self.session.flush()

        # Replace specifications.
        self.session.query(Specification).filter(
            Specification.phone_id == phone.id
        ).delete(synchronize_session=False)

        seen: set[tuple[int, str, int]] = set()
        for category_name, key, value, position in scraped.specs:
            category_id = self._category_id(category_name)
            identity = (category_id, key, position)
            if identity in seen:
                continue
            seen.add(identity)
            self.session.add(
                Specification(
                    phone_id=phone.id,
                    category_id=category_id,
                    spec_key=key,
                    spec_value=value,
                    position=position,
                )
            )

        # Replace prices.
        self.session.query(Price).filter(Price.phone_id == phone.id).delete(
            synchronize_session=False
        )
        for currency, amount, raw_text in scraped.prices:
            self.session.add(
                Price(
                    phone_id=phone.id,
                    currency=currency,
                    amount=amount,
                    raw_text=raw_text,
                )
            )

        self.session.flush()
        return phone

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def _base_query(self):
        return select(Phone).options(
            selectinload(Phone.specifications).selectinload(Specification.category),
            selectinload(Phone.prices),
        )

    def list_phones(self, limit: int | None = None, offset: int = 0) -> list[Phone]:
        """Return phones ordered newest-first, then alphabetically."""
        query = self._base_query().order_by(*_newest_first())
        if offset:
            query = query.offset(offset)
        if limit is not None:
            query = query.limit(limit)
        return list(self.session.scalars(query).unique())

    def count_phones(self) -> int:
        return self.session.scalar(select(func.count(Phone.id))) or 0

    def count_specifications(self) -> int:
        return self.session.scalar(select(func.count(Specification.id))) or 0

    def count_prices(self) -> int:
        return self.session.scalar(select(func.count(Price.id))) or 0

    def get_by_id(self, phone_id: int) -> Phone | None:
        return self.session.scalar(self._base_query().where(Phone.id == phone_id))

    def get_by_slug(self, slug: str) -> Phone | None:
        return self.session.scalar(self._base_query().where(Phone.slug == slug))

    def find_by_name(self, name: str) -> Phone | None:
        """Resolve a phone from a loosely-typed model name.

        Tries an exact match first, then a case-insensitive ``LIKE``.  This is
        what lets a user write "s23 ultra" and still reach
        "Samsung Galaxy S23 Ultra".
        """
        cleaned = " ".join(name.split())
        if not cleaned:
            return None

        exact = self.session.scalar(
            self._base_query().where(func.lower(Phone.name) == cleaned.casefold())
        )
        if exact is not None:
            return exact

        return self.session.scalar(
            self._base_query()
            .where(Phone.name.ilike(f"%{cleaned}%"))
            .order_by(func.char_length(Phone.name))
        )

    def search_by_name(self, fragment: str, limit: int = 10) -> list[Phone]:
        """Return every phone whose name contains ``fragment``."""
        query = (
            self._base_query()
            .where(Phone.name.ilike(f"%{fragment.strip()}%"))
            .order_by(*_newest_first())
            .limit(limit)
        )
        return list(self.session.scalars(query).unique())

    def top_by_column(
        self, column_name: str, limit: int = 5, descending: bool = True
    ) -> list[Phone]:
        """Rank phones by any numeric column, e.g. ``battery_capacity_mah``.

        This backs superlative questions ("best battery life", "biggest
        screen") with a deterministic SQL answer rather than an LLM guess.
        """
        column = getattr(Phone, column_name, None)
        if column is None:
            raise ValueError(f"Unknown ranking column: {column_name!r}")

        order = column.desc() if descending else column.asc()
        query = (
            self._base_query()
            .where(column.is_not(None))
            .order_by(order)
            .limit(limit)
        )
        return list(self.session.scalars(query).unique())

    def get_all_names(self) -> list[str]:
        return list(
            self.session.scalars(select(Phone.name).order_by(Phone.name)).all()
        )

    def specs_by_category(self, phone: Phone) -> dict[str, list[tuple[str, str]]]:
        """Group a phone's specifications by category, preserving page order."""
        grouped: dict[str, list[tuple[str, str]]] = {}
        ordered = sorted(
            phone.specifications,
            key=lambda s: (s.category.display_order, s.position),
        )
        for spec in ordered:
            grouped.setdefault(spec.category.name, []).append(
                (spec.spec_key, spec.spec_value)
            )
        return grouped

    def statistics(self) -> dict[str, Any]:
        """Aggregate counts and ranges, used by the API's ``/stats`` endpoint."""
        return {
            "phones": self.count_phones(),
            "specifications": self.count_specifications(),
            "prices": self.count_prices(),
            "release_years": [
                year
                for (year,) in self.session.execute(
                    select(Phone.release_year)
                    .where(Phone.release_year.is_not(None))
                    .distinct()
                    .order_by(Phone.release_year)
                ).all()
            ],
            "largest_battery_mah": self.session.scalar(
                select(func.max(Phone.battery_capacity_mah))
            ),
            "largest_display_inches": self.session.scalar(
                select(func.max(Phone.display_size_inches))
            ),
        }

    def rank_by_battery_life(self, limit: int = 6) -> list[Phone]:
        """Rank phones by measured battery endurance rather than capacity.

        Restricted to GSMArena's current "active use score" because their older
        "endurance rating" is on a different scale - mixing the two would
        produce a meaningless ordering.
        """
        query = (
            self._base_query()
            .where(Phone.battery_endurance_metric == "active_use_score")
            .where(Phone.battery_endurance_hours.is_not(None))
            .order_by(Phone.battery_endurance_hours.desc())
            .limit(limit)
        )
        return list(self.session.scalars(query).unique())

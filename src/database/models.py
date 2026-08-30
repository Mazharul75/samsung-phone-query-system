"""SQLAlchemy ORM models describing the Samsung phone database.

Schema overview
---------------
phones
    One row per phone model. Holds identity fields plus the *normalised*
    headline specifications (numeric columns such as ``battery_capacity_mah``
    or ``display_size_inches``). Keeping these parsed values in typed columns
    is what allows the system to answer superlative and comparison questions
    with plain SQL - e.g. "which Samsung phone has the best battery life?".

spec_categories
    Lookup table for the specification groups GSMArena publishes
    (Network, Launch, Body, Display, Platform, Memory, Main Camera, ...).

specifications
    Full-fidelity key/value rows. Every specification found on the source page
    is stored here, so nothing from the original listing is lost even when it
    has no dedicated column on ``phones``.

prices
    Currency-aware price points parsed from the listing, kept separate because
    a phone can carry several prices (EUR / USD / INR) and prices change over
    time.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    DECIMAL,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    """Timezone-aware UTC timestamp (datetime.utcnow is deprecated)."""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Declarative base shared by every table in the schema."""


class Phone(Base):
    """A single Samsung phone model."""

    __tablename__ = "phones"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # -- Identity ------------------------------------------------------
    name: Mapped[str] = mapped_column(String(150), nullable=False, unique=True)
    slug: Mapped[str] = mapped_column(String(150), nullable=False, unique=True)
    brand: Mapped[str] = mapped_column(String(50), nullable=False, default="Samsung")
    source_url: Mapped[str] = mapped_column(String(500), nullable=False)
    image_url: Mapped[str | None] = mapped_column(String(500))

    # -- Launch --------------------------------------------------------
    announced: Mapped[str | None] = mapped_column(String(120))
    release_status: Mapped[str | None] = mapped_column(String(200))
    release_year: Mapped[int | None] = mapped_column(Integer, index=True)

    # -- Body ----------------------------------------------------------
    dimensions: Mapped[str | None] = mapped_column(String(200))
    weight_grams: Mapped[float | None] = mapped_column(DECIMAL(6, 1))
    build: Mapped[str | None] = mapped_column(String(400))

    # -- Display -------------------------------------------------------
    display_type: Mapped[str | None] = mapped_column(String(300))
    display_size_inches: Mapped[float | None] = mapped_column(DECIMAL(4, 2), index=True)
    display_resolution: Mapped[str | None] = mapped_column(String(200))
    refresh_rate_hz: Mapped[int | None] = mapped_column(Integer)
    display_protection: Mapped[str | None] = mapped_column(String(200))

    # -- Platform ------------------------------------------------------
    operating_system: Mapped[str | None] = mapped_column(String(300))
    chipset: Mapped[str | None] = mapped_column(String(250), index=True)
    cpu: Mapped[str | None] = mapped_column(String(400))
    gpu: Mapped[str | None] = mapped_column(String(200))

    # -- Memory --------------------------------------------------------
    memory_internal: Mapped[str | None] = mapped_column(String(400))
    max_ram_gb: Mapped[int | None] = mapped_column(Integer, index=True)
    max_storage_gb: Mapped[int | None] = mapped_column(Integer, index=True)
    card_slot: Mapped[str | None] = mapped_column(String(150))

    # -- Cameras -------------------------------------------------------
    main_camera_summary: Mapped[str | None] = mapped_column(Text)
    main_camera_mp: Mapped[float | None] = mapped_column(DECIMAL(6, 1), index=True)
    main_camera_video: Mapped[str | None] = mapped_column(String(400))
    selfie_camera_summary: Mapped[str | None] = mapped_column(Text)
    selfie_camera_mp: Mapped[float | None] = mapped_column(DECIMAL(6, 1))
    selfie_camera_video: Mapped[str | None] = mapped_column(String(400))

    # -- Battery -------------------------------------------------------
    battery_type: Mapped[str | None] = mapped_column(String(200))
    battery_capacity_mah: Mapped[int | None] = mapped_column(Integer, index=True)
    charging: Mapped[str | None] = mapped_column(String(400))
    charging_watts: Mapped[int | None] = mapped_column(Integer)
    #: GSMArena's battery test result, verbatim (e.g. "Active use score 14:49h").
    battery_endurance: Mapped[str | None] = mapped_column(String(120))
    #: The same result as a number, so battery life can be ranked in SQL.
    battery_endurance_hours: Mapped[float | None] = mapped_column(
        DECIMAL(6, 2), index=True
    )
    #: Which test produced it. GSMArena changed methodology, and the two
    #: scales are not comparable, so rankings must stay within one metric.
    battery_endurance_metric: Mapped[str | None] = mapped_column(String(40), index=True)

    # -- Misc ----------------------------------------------------------
    colors: Mapped[str | None] = mapped_column(String(500))
    network_technology: Mapped[str | None] = mapped_column(String(250))
    sim: Mapped[str | None] = mapped_column(String(300))

    # -- Bookkeeping ---------------------------------------------------
    scraped_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow
    )

    specifications: Mapped[list["Specification"]] = relationship(
        back_populates="phone",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    prices: Mapped[list["Price"]] = relationship(
        back_populates="phone",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"<Phone id={self.id} name={self.name!r}>"


class SpecCategory(Base):
    """A specification group such as Display or Battery."""

    __tablename__ = "spec_categories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(80), nullable=False, unique=True)
    display_order: Mapped[int] = mapped_column(Integer, default=0)

    specifications: Mapped[list["Specification"]] = relationship(
        back_populates="category"
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"<SpecCategory {self.name!r}>"


class Specification(Base):
    """A single key/value specification row belonging to one phone."""

    __tablename__ = "specifications"
    __table_args__ = (
        UniqueConstraint(
            "phone_id", "category_id", "spec_key", "position", name="uq_spec_row"
        ),
        Index("ix_spec_key", "spec_key"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    phone_id: Mapped[int] = mapped_column(
        ForeignKey("phones.id", ondelete="CASCADE"), nullable=False, index=True
    )
    category_id: Mapped[int] = mapped_column(
        ForeignKey("spec_categories.id"), nullable=False, index=True
    )
    spec_key: Mapped[str] = mapped_column(String(120), nullable=False)
    spec_value: Mapped[str] = mapped_column(Text, nullable=False)
    #: Preserves the original ordering of rows inside a category and
    #: disambiguates repeated keys (e.g. two "4G bands" rows).
    position: Mapped[int] = mapped_column(Integer, default=0)

    phone: Mapped["Phone"] = relationship(back_populates="specifications")
    category: Mapped["SpecCategory"] = relationship(back_populates="specifications")

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"<Specification {self.spec_key!r}>"


class Price(Base):
    """A price point for a phone in one currency."""

    __tablename__ = "prices"
    __table_args__ = (
        UniqueConstraint("phone_id", "currency", name="uq_price_currency"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    phone_id: Mapped[int] = mapped_column(
        ForeignKey("phones.id", ondelete="CASCADE"), nullable=False, index=True
    )
    currency: Mapped[str] = mapped_column(String(10), nullable=False)
    amount: Mapped[float] = mapped_column(DECIMAL(12, 2), nullable=False)
    raw_text: Mapped[str | None] = mapped_column(String(300))
    source: Mapped[str] = mapped_column(String(50), default="gsmarena")
    captured_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    phone: Mapped["Phone"] = relationship(back_populates="prices")

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"<Price {self.amount} {self.currency}>"

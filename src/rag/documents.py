"""Turns database rows into the text documents the retriever searches.

Chunking strategy
-----------------
Rather than dumping one huge blob per phone, each phone is split into several
focused documents:

* **One overview document** carrying the headline specifications.  This is what
  matches broad questions such as "tell me about the S24 Ultra".
* **One document per specification category** (Display, Platform, Battery, ...).
  A question about cameras then retrieves the camera document instead of a
  paragraph where the camera details are buried among network bands.

Every document repeats the phone name in its text.  That redundancy is
deliberate: it keeps each chunk independently meaningful, so a retrieved
snippet never leaves the model guessing which phone it describes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.database.models import Phone

#: Categories that carry little conversational value; excluded to keep the
#: index focused.  Their content stays in the database and remains reachable
#: through the API's specification endpoints.
_SKIPPED_CATEGORIES = {"Network", "EU LABEL"}


@dataclass
class Document:
    """A retrievable chunk of text plus the metadata used for filtering."""

    doc_id: str
    text: str
    phone_id: int
    phone_name: str
    section: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.text)


def _describe_lenses(summary: str | None) -> str:
    """Render a bullet-joined camera cell as explicitly numbered lenses."""
    if not summary:
        return ""
    lenses = [part.strip() for part in summary.split(" | ") if part.strip()]
    if len(lenses) <= 1:
        return summary
    return " ".join(
        f"Lens {index}: {lens}." for index, lens in enumerate(lenses, start=1)
    )


def _format_price(phone: Phone) -> str:
    if not phone.prices:
        return "Price: not listed."
    parts = [f"{p.currency} {float(p.amount):,.2f}" for p in sorted(
        phone.prices, key=lambda p: p.currency
    )]
    return "Approximate launch price: " + ", ".join(parts) + "."


def build_overview_text(phone: Phone) -> str:
    """Compose the headline description used for broad questions."""
    lines: list[str] = [f"{phone.name} - overview."]

    if phone.announced:
        lines.append(f"Announced: {phone.announced}.")
    if phone.release_status:
        lines.append(f"Availability: {phone.release_status}.")
    if phone.display_size_inches:
        display = f"Display: {phone.display_size_inches} inch"
        if phone.display_type:
            display += f" {phone.display_type}"
        if phone.refresh_rate_hz:
            display += f", {phone.refresh_rate_hz}Hz refresh rate"
        if phone.display_resolution:
            display += f", resolution {phone.display_resolution}"
        lines.append(display + ".")
    if phone.chipset:
        processor = f"Processor: {phone.chipset}"
        if phone.cpu:
            processor += f"; CPU {phone.cpu}"
        if phone.gpu:
            processor += f"; GPU {phone.gpu}"
        lines.append(processor + ".")
    if phone.max_ram_gb or phone.max_storage_gb:
        lines.append(
            f"Memory: up to {phone.max_ram_gb or '?'}GB RAM and "
            f"{phone.max_storage_gb or '?'}GB storage. "
            f"Configurations: {phone.memory_internal or 'not listed'}."
        )
    if phone.main_camera_mp:
        lines.append(
            f"Main camera: {phone.main_camera_mp:g} MP primary sensor. "
            + _describe_lenses(phone.main_camera_summary)
        )
    if phone.selfie_camera_mp:
        lines.append(f"Selfie camera: {phone.selfie_camera_mp:g} MP.")
    if phone.battery_capacity_mah:
        battery = f"Battery: {phone.battery_capacity_mah} mAh"
        if phone.charging_watts:
            battery += f" with {phone.charging_watts}W charging"
        if phone.battery_endurance:
            battery += f"; GSMArena battery test result: {phone.battery_endurance}"
        lines.append(battery + ".")
    if phone.weight_grams:
        lines.append(
            f"Body: {phone.dimensions or 'dimensions not listed'}, "
            f"{float(phone.weight_grams):g} g."
        )
    if phone.operating_system:
        lines.append(f"Software: {phone.operating_system}.")
    if phone.colors:
        lines.append(f"Colours: {phone.colors}.")

    lines.append(_format_price(phone))
    return " ".join(lines)


#: Categories whose values list several independent items in one cell.
_MULTI_ITEM_CATEGORIES = ("Main Camera", "Selfie camera")


def _format_spec_row(category: str, key: str, value: str) -> str:
    """Render one specification row for inclusion in a document.

    GSMArena packs every camera lens into a single cell separated by bullets,
    which the scraper normalises to ``|``.  Left as one run-on string, the
    language model reliably merges two lenses into one or repeats a lens.
    Splitting the cell into explicitly numbered lenses removes the ambiguity.
    """
    if category.startswith(_MULTI_ITEM_CATEGORIES) and " | " in value:
        lenses = [part.strip() for part in value.split(" | ") if part.strip()]
        if len(lenses) > 1:
            numbered = " ".join(
                f"Lens {index}: {lens}." for index, lens in enumerate(lenses, start=1)
            )
            return f"{key} ({len(lenses)} lenses): {numbered}"

    return f"{key}: {value}."


def build_documents_for_phone(
    phone: Phone, specs_by_category: dict[str, list[tuple[str, str]]]
) -> list[Document]:
    """Build the overview plus per-category documents for one phone."""
    documents: list[Document] = [
        Document(
            doc_id=f"{phone.slug}::overview",
            text=build_overview_text(phone),
            phone_id=phone.id,
            phone_name=phone.name,
            section="Overview",
            metadata={
                "release_year": phone.release_year,
                "chipset": phone.chipset,
                "battery_capacity_mah": phone.battery_capacity_mah,
            },
        )
    ]

    for category, rows in specs_by_category.items():
        if category in _SKIPPED_CATEGORIES:
            continue

        details = " ".join(
            _format_spec_row(category, key, value) for key, value in rows
        )
        text = f"{phone.name} - {category} specifications. {details}"
        documents.append(
            Document(
                doc_id=f"{phone.slug}::{category.lower().replace(' ', '-')}",
                text=text,
                phone_id=phone.id,
                phone_name=phone.name,
                section=category,
                metadata={"release_year": phone.release_year},
            )
        )

    return documents


def build_corpus(repository: Any) -> list[Document]:
    """Build the full document corpus from every phone in the database."""
    documents: list[Document] = []
    for phone in repository.list_phones():
        grouped = repository.specs_by_category(phone)
        documents.extend(build_documents_for_phone(phone, grouped))
    return documents

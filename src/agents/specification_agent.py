"""Specification Agent - retrieves a phone's full technical dossier.

This is the data-gathering specialist described in the brief: it locates the
phone in the database and assembles every specification the review will need.

It deliberately contains no language model.  Facts are read straight from
MySQL, so the figures a review quotes are exactly what was scraped rather than
whatever the model recalls about Samsung phones.
"""

from __future__ import annotations

from typing import Any

from src.agents.base import Agent, AgentContext
from src.database.connection import session_scope
from src.database.repository import PhoneRepository

#: Specification groups worth putting in front of the review writer.  Network
#: band lists are omitted - they are long and add nothing to a review.
_REVIEW_CATEGORIES = (
    "Launch",
    "Body",
    "Display",
    "Platform",
    "Memory",
    "Main Camera",
    "Selfie camera",
    "Sound",
    "Battery",
    "Our Tests",
    "Misc",
)


class SpecificationAgent(Agent):
    """Fetches structured specifications for one phone from the database."""

    name = "specification_agent"
    role = "Technical data retrieval specialist"
    goal = (
        "Locate the requested phone in the database and assemble a complete, "
        "accurate specification dossier for downstream agents."
    )

    def execute(self, context: AgentContext) -> dict[str, Any]:
        with session_scope() as session:
            repository = PhoneRepository(session)
            phone = repository.find_by_name(context.phone_name)
            if phone is None:
                available = ", ".join(repository.get_all_names()[:8])
                raise LookupError(
                    f"No phone matching {context.phone_name!r} is in the database. "
                    f"Known models include: {available}..."
                )

            grouped = repository.specs_by_category(phone)
            dossier: dict[str, Any] = {
                "id": phone.id,
                "name": phone.name,
                "slug": phone.slug,
                "brand": phone.brand,
                "source_url": phone.source_url,
                "image_url": phone.image_url,
                "release_year": phone.release_year,
                "announced": phone.announced,
                "release_status": phone.release_status,
                "headline": {
                    "chipset": phone.chipset,
                    "cpu": phone.cpu,
                    "gpu": phone.gpu,
                    "operating_system": phone.operating_system,
                    "display_size_inches": _number(phone.display_size_inches),
                    "display_type": phone.display_type,
                    "display_resolution": phone.display_resolution,
                    "refresh_rate_hz": phone.refresh_rate_hz,
                    "max_ram_gb": phone.max_ram_gb,
                    "max_storage_gb": phone.max_storage_gb,
                    "memory_internal": phone.memory_internal,
                    "card_slot": phone.card_slot,
                    "main_camera_mp": _number(phone.main_camera_mp),
                    "main_camera_summary": phone.main_camera_summary,
                    "main_camera_video": phone.main_camera_video,
                    "selfie_camera_mp": _number(phone.selfie_camera_mp),
                    "battery_capacity_mah": phone.battery_capacity_mah,
                    "charging_watts": phone.charging_watts,
                    "battery_endurance": phone.battery_endurance,
                    "weight_grams": _number(phone.weight_grams),
                    "dimensions": phone.dimensions,
                    "build": phone.build,
                    "colors": phone.colors,
                },
                "prices": [
                    {
                        "currency": price.currency,
                        "amount": float(price.amount),
                    }
                    for price in sorted(phone.prices, key=lambda p: p.currency)
                ],
                "specifications": {
                    category: rows
                    for category, rows in grouped.items()
                    if category in _REVIEW_CATEGORIES
                },
                "specification_count": len(phone.specifications),
            }

        context.data["specifications"] = dossier
        context.phone_name = dossier["name"]
        return dossier

    def describe(self, output: dict[str, Any]) -> str:
        return (
            f"Retrieved {output['specification_count']} specifications for "
            f"{output['name']} across {len(output['specifications'])} categories."
        )


def _number(value: Any) -> float | None:
    """Convert a SQL ``DECIMAL`` to a plain float for JSON serialisation."""
    return None if value is None else float(value)

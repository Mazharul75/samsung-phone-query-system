"""Comparison Agent - positions one phone against the rest of the catalogue.

A specification sheet on its own does not make a review: "5000 mAh" only means
something next to what the other phones offer.  This agent supplies that
missing half by computing, entirely in SQL, where the phone ranks on each key
attribute and which models are its closest alternatives.

Like the Specification Agent it uses no language model - every claim it emits
("2nd largest battery of 15 phones") is arithmetic over the database, so the
review writer can state it without risk of inventing a ranking.
"""

from __future__ import annotations

from typing import Any

from src.agents.base import Agent, AgentContext
from src.database.connection import session_scope
from src.database.repository import PhoneRepository

#: Attributes the review is positioned on, and whether higher is better.
_RANKED_ATTRIBUTES: tuple[tuple[str, str, bool], ...] = (
    ("battery_capacity_mah", "battery capacity", True),
    ("charging_watts", "charging speed", True),
    ("display_size_inches", "display size", True),
    ("main_camera_mp", "main camera resolution", True),
    ("selfie_camera_mp", "selfie camera resolution", True),
    ("max_ram_gb", "RAM", True),
    ("max_storage_gb", "storage", True),
    ("weight_grams", "weight", False),
)


class ComparisonAgent(Agent):
    """Ranks the phone against the catalogue and finds its nearest rivals."""

    name = "comparison_agent"
    role = "Competitive positioning analyst"
    goal = (
        "Establish how the phone ranks against every other model in the "
        "database so the review can put its specifications in context."
    )

    def execute(self, context: AgentContext) -> dict[str, Any]:
        dossier = context.data.get("specifications")
        if not dossier:
            raise RuntimeError(
                "ComparisonAgent requires the SpecificationAgent to run first."
            )

        target_id = dossier["id"]

        with session_scope() as session:
            repository = PhoneRepository(session)
            phones = repository.list_phones()
            total = len(phones)

            rankings: list[dict[str, Any]] = []
            for attribute, label, higher_is_better in _RANKED_ATTRIBUTES:
                scored = [
                    (phone, float(getattr(phone, attribute)))
                    for phone in phones
                    if getattr(phone, attribute) is not None
                ]
                if len(scored) < 2:
                    continue

                scored.sort(key=lambda item: item[1], reverse=higher_is_better)
                position = next(
                    (
                        index
                        for index, (phone, _value) in enumerate(scored, start=1)
                        if phone.id == target_id
                    ),
                    None,
                )
                if position is None:
                    continue

                value = next(v for phone, v in scored if phone.id == target_id)
                best_phone, best_value = scored[0]
                rankings.append(
                    {
                        "attribute": label,
                        "value": value,
                        "rank": position,
                        "out_of": len(scored),
                        "is_best": position == 1,
                        "leader": best_phone.name,
                        "leader_value": best_value,
                        "higher_is_better": higher_is_better,
                    }
                )

            target = repository.get_by_id(target_id)
            rivals = _closest_rivals(target, phones)
            rival_summaries = [
                {
                    "name": rival.name,
                    "release_year": rival.release_year,
                    "chipset": rival.chipset,
                    "battery_capacity_mah": rival.battery_capacity_mah,
                    "main_camera_mp": _number(rival.main_camera_mp),
                    "display_size_inches": _number(rival.display_size_inches),
                }
                for rival in rivals
            ]

        strengths = [r for r in rankings if r["rank"] <= 3]
        weaknesses = [
            r for r in rankings if r["rank"] > max(3, int(r["out_of"] * 0.6))
        ]

        analysis = {
            "catalogue_size": total,
            "rankings": rankings,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "closest_rivals": rival_summaries,
        }
        context.data["comparison"] = analysis
        return analysis

    def describe(self, output: dict[str, Any]) -> str:
        return (
            f"Ranked against {output['catalogue_size']} phones on "
            f"{len(output['rankings'])} attributes; found "
            f"{len(output['strengths'])} standout strengths and "
            f"{len(output['closest_rivals'])} close rivals."
        )


def _closest_rivals(target: Any, phones: list[Any], limit: int = 3) -> list[Any]:
    """Find the models most similar to ``target``.

    Similarity is a simple distance over the attributes buyers actually
    cross-shop on - release year, screen size, battery and camera - each scaled
    so no single attribute dominates the total.
    """
    weights = (
        ("release_year", 1.0),
        ("display_size_inches", 2.0),
        ("battery_capacity_mah", 0.002),
        ("main_camera_mp", 0.02),
    )

    scored: list[tuple[float, Any]] = []
    for phone in phones:
        if phone.id == target.id:
            continue

        distance = 0.0
        comparable = 0
        for attribute, weight in weights:
            left, right = getattr(target, attribute), getattr(phone, attribute)
            if left is None or right is None:
                continue
            distance += abs(float(left) - float(right)) * weight
            comparable += 1

        if comparable:
            scored.append((distance / comparable, phone))

    scored.sort(key=lambda item: item[0])
    return [phone for _distance, phone in scored[:limit]]


def _number(value: Any) -> float | None:
    return None if value is None else float(value)

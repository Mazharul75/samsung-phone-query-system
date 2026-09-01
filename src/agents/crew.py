"""The review crew - orchestrates the specialist agents.

Pipeline
--------
1. :class:`~src.agents.specification_agent.SpecificationAgent` reads the phone's
   full specification dossier from MySQL.
2. :class:`~src.agents.comparison_agent.ComparisonAgent` ranks it against every
   other phone in the catalogue.
3. :class:`~src.agents.review_agent.ReviewAgent` writes the review from what the
   first two produced.

The agents communicate only through the shared :class:`AgentContext`, so each
can be tested on its own and the order is easy to change.  The crew enforces
the dependencies: without specifications there is nothing to review, so that
failure stops the run, whereas a failed comparison only costs the review its
competitive context and the run continues.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from src.agents.base import Agent, AgentContext, AgentResult
from src.agents.comparison_agent import ComparisonAgent
from src.agents.review_agent import ReviewAgent
from src.agents.specification_agent import SpecificationAgent

logger = logging.getLogger(__name__)


@dataclass
class CrewResult:
    """Everything one crew run produced."""

    phone: str
    success: bool
    review: dict[str, Any] | None = None
    specifications: dict[str, Any] | None = None
    comparison: dict[str, Any] | None = None
    transcript: list[AgentResult] = field(default_factory=list)
    duration_seconds: float = 0.0
    error: str | None = None

    def to_dict(self, include_details: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "phone": self.phone,
            "success": self.success,
            "duration_seconds": round(self.duration_seconds, 2),
            "agents": [result.to_dict() for result in self.transcript],
            "error": self.error,
        }
        if self.review:
            payload["review"] = self.review
        if include_details:
            payload["specifications"] = self.specifications
            payload["comparison"] = self.comparison
        return payload


class ReviewCrew:
    """Runs the specification, comparison and review agents in order."""

    def __init__(self, agents: list[Agent] | None = None) -> None:
        self.agents: list[Agent] = agents or [
            SpecificationAgent(),
            ComparisonAgent(),
            ReviewAgent(),
        ]

    #: Agents whose failure makes the rest of the pipeline pointless.
    REQUIRED = {"specification_agent", "review_agent"}

    def run(self, phone_name: str) -> CrewResult:
        """Produce a full review for ``phone_name``."""
        started = time.monotonic()
        context = AgentContext(phone_name=phone_name)
        logger.info("Crew starting for %r with %d agents", phone_name, len(self.agents))

        error: str | None = None
        for agent in self.agents:
            result = agent.run(context)
            if not result.success and agent.name in self.REQUIRED:
                error = f"{agent.name} failed: {result.error}"
                logger.error("Crew aborted - %s", error)
                break
            if not result.success:
                logger.warning(
                    "%s failed but is optional; continuing without it.", agent.name
                )

        review = context.data.get("review")
        return CrewResult(
            phone=context.phone_name,
            success=error is None and review is not None,
            review=review,
            specifications=context.data.get("specifications"),
            comparison=context.data.get("comparison"),
            transcript=context.transcript,
            duration_seconds=time.monotonic() - started,
            error=error,
        )

    def describe(self) -> list[dict[str, str]]:
        """Describe the crew's members, used by the API's ``/agents`` endpoint."""
        return [
            {"name": agent.name, "role": agent.role, "goal": agent.goal}
            for agent in self.agents
        ]


_crew: ReviewCrew | None = None


def get_crew() -> ReviewCrew:
    """Return the process-wide crew singleton."""
    global _crew
    if _crew is None:
        _crew = ReviewCrew()
    return _crew

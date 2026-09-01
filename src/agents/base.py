"""Shared foundation for the specialist agents.

Every agent declares a name, a role and a goal, then implements :meth:`run`.
Each returns an :class:`AgentResult` rather than a bare string, so the
orchestrator can record who produced what, how long it took, and whether the
step succeeded.  That transcript is what makes the collaboration inspectable
instead of a black box - the API exposes it alongside the finished review.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """The outcome of a single agent's turn."""

    agent: str
    role: str
    success: bool
    output: Any = None
    summary: str = ""
    duration_seconds: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "role": self.role,
            "success": self.success,
            "summary": self.summary,
            "duration_seconds": round(self.duration_seconds, 2),
            "error": self.error,
        }


@dataclass
class AgentContext:
    """The shared blackboard the agents read from and write to.

    Agents never call one another directly.  Each reads what earlier agents
    deposited here and adds its own contribution, which keeps them independent
    and individually testable.
    """

    phone_name: str
    data: dict[str, Any] = field(default_factory=dict)
    transcript: list[AgentResult] = field(default_factory=list)

    def record(self, result: AgentResult) -> None:
        self.transcript.append(result)

    @property
    def succeeded(self) -> bool:
        return all(result.success for result in self.transcript)


class Agent(ABC):
    """Base class for every specialist agent."""

    #: Short identifier used in the transcript.
    name: str = "agent"
    #: What this agent is responsible for.
    role: str = ""
    #: What a successful turn looks like.
    goal: str = ""

    @abstractmethod
    def execute(self, context: AgentContext) -> Any:
        """Do the agent's work and return its contribution."""

    def run(self, context: AgentContext) -> AgentResult:
        """Execute the agent, timing it and capturing any failure.

        A failing agent must not abort the whole crew - the orchestrator
        decides whether the pipeline can continue without this step.
        """
        logger.info("[%s] starting", self.name)
        started = time.monotonic()
        try:
            output = self.execute(context)
            result = AgentResult(
                agent=self.name,
                role=self.role,
                success=True,
                output=output,
                summary=self.describe(output),
                duration_seconds=time.monotonic() - started,
            )
        except Exception as exc:
            logger.exception("[%s] failed", self.name)
            result = AgentResult(
                agent=self.name,
                role=self.role,
                success=False,
                duration_seconds=time.monotonic() - started,
                error=str(exc),
            )

        logger.info(
            "[%s] finished in %.2fs (success=%s)",
            self.name,
            result.duration_seconds,
            result.success,
        )
        context.record(result)
        return result

    def describe(self, output: Any) -> str:
        """One-line description of what this turn produced."""
        return "completed"

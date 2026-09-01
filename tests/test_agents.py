"""Tests for the multi-agent review system."""

from __future__ import annotations

import pytest

from src.agents.base import Agent, AgentContext, AgentResult
from src.agents.comparison_agent import ComparisonAgent
from src.agents.crew import ReviewCrew
from src.agents.review_agent import (
    _card_slot_note,
    summarise_cpu,
    trim_to_last_sentence,
)
from src.agents.specification_agent import SpecificationAgent


# ---------------------------------------------------------------------------
# Base machinery
# ---------------------------------------------------------------------------
class _OkAgent(Agent):
    name = "ok_agent"
    role = "test"

    def execute(self, context: AgentContext) -> str:
        context.data["ok"] = True
        return "done"


class _FailingAgent(Agent):
    name = "specification_agent"  # a required member of the crew
    role = "test"

    def execute(self, context: AgentContext) -> str:
        raise RuntimeError("boom")


class TestAgentBase:
    def test_successful_run_is_recorded(self):
        context = AgentContext(phone_name="Galaxy S23")
        result = _OkAgent().run(context)

        assert result.success
        assert result.output == "done"
        assert result.duration_seconds >= 0
        assert context.transcript == [result]

    def test_failure_is_captured_not_raised(self):
        context = AgentContext(phone_name="Galaxy S23")
        result = _FailingAgent().run(context)

        assert not result.success
        assert "boom" in result.error
        assert len(context.transcript) == 1

    def test_result_serialises(self):
        payload = AgentResult(
            agent="a", role="r", success=True, summary="s"
        ).to_dict()
        assert set(payload) == {
            "agent", "role", "success", "summary", "duration_seconds", "error"
        }


class TestCpuSummary:
    """The raw CPU string is never shown to the model - see summarise_cpu."""

    def test_digit_core_count(self):
        cpu = (
            "8-core (1x3.39GHz Cortex-X4 & 3x3.1GHz Cortex-A720 & "
            "2x2.9GHz Cortex-A720 & 2x2.2GHz Cortex-A520)"
        )
        assert summarise_cpu(cpu) == (
            "8 cores, peak clock 3.39 GHz, fastest core Cortex-X4"
        )

    def test_word_core_count(self):
        cpu = "Octa-core (1x3.36 GHz Cortex-X3 & 3x2.0 GHz Cortex-A510)"
        assert summarise_cpu(cpu).startswith("8 cores")

    def test_peak_clock_is_the_maximum_not_the_first(self):
        cpu = "Octa-core (4x1.8 GHz Cortex-A55 & 1x2.9 GHz Cortex-X1)"
        assert "peak clock 2.9 GHz" in summarise_cpu(cpu)

    def test_custom_cores_omit_the_core_name_gracefully(self):
        cpu = "8-core (2x4.47 GHz & 6x3.53 GHz)"
        summary = summarise_cpu(cpu)
        assert summary == "8 cores, peak clock 4.47 GHz"

    def test_handles_missing_input(self):
        assert summarise_cpu(None) is None
        assert summarise_cpu("") is None


class TestCardSlotNote:
    def test_absent_slot_is_stated_explicitly(self):
        assert "no microSD" in _card_slot_note("No")

    def test_present_slot_is_passed_through(self):
        value = "microSDXC (uses shared SIM slot)"
        assert _card_slot_note(value) == value

    def test_handles_missing_input(self):
        assert _card_slot_note(None) is None


class TestSentenceTrimming:
    def test_removes_trailing_fragment(self):
        text = "This sentence is complete. This one was cut off mid"
        assert trim_to_last_sentence(text) == "This sentence is complete."

    def test_keeps_complete_text_untouched(self):
        text = "Everything here is complete."
        assert trim_to_last_sentence(text) == text

    def test_keeps_short_fragment_rather_than_emptying(self):
        assert trim_to_last_sentence("no terminator here") == "no terminator here"

    def test_handles_empty_input(self):
        assert trim_to_last_sentence("   ") == ""


# ---------------------------------------------------------------------------
# Data agents (no language model - fast and deterministic)
# ---------------------------------------------------------------------------
class TestSpecificationAgent:
    def test_builds_a_complete_dossier(self, populated_repository):
        context = AgentContext(phone_name="Galaxy S23 Ultra")
        result = SpecificationAgent().run(context)

        assert result.success
        dossier = result.output
        assert "S23 Ultra" in dossier["name"]
        assert dossier["specification_count"] > 30
        assert dossier["headline"]["battery_capacity_mah"] == 5000
        assert dossier["headline"]["main_camera_mp"] == 200.0
        assert dossier["specifications"]

    def test_unknown_phone_fails_cleanly(self, populated_repository):
        context = AgentContext(phone_name="Nokia 3310")
        result = SpecificationAgent().run(context)

        assert not result.success
        assert "No phone matching" in result.error

    def test_writes_into_the_shared_context(self, populated_repository):
        context = AgentContext(phone_name="Galaxy S23")
        SpecificationAgent().run(context)
        assert "specifications" in context.data


class TestComparisonAgent:
    @pytest.fixture()
    def analysed(self, populated_repository):
        context = AgentContext(phone_name="Galaxy S24 Ultra")
        SpecificationAgent().run(context)
        result = ComparisonAgent().run(context)
        assert result.success
        return result.output

    def test_ranks_across_the_catalogue(self, analysed):
        assert analysed["catalogue_size"] >= 10
        assert analysed["rankings"]

    def test_rankings_are_internally_consistent(self, analysed):
        for item in analysed["rankings"]:
            assert 1 <= item["rank"] <= item["out_of"]
            assert item["is_best"] == (item["rank"] == 1)

    def test_finds_close_rivals_excluding_itself(self, analysed):
        rivals = analysed["closest_rivals"]
        assert rivals
        assert all("S24 Ultra" not in rival["name"] for rival in rivals)

    def test_requires_the_specification_agent_first(self, populated_repository):
        context = AgentContext(phone_name="Galaxy S23")
        result = ComparisonAgent().run(context)
        assert not result.success
        assert "SpecificationAgent" in result.error


# ---------------------------------------------------------------------------
# The crew
# ---------------------------------------------------------------------------
class TestCrewComposition:
    def test_default_crew_has_the_three_specialists(self):
        names = [agent.name for agent in ReviewCrew().agents]
        assert names == ["specification_agent", "comparison_agent", "review_agent"]

    def test_describe_reports_roles_and_goals(self):
        described = ReviewCrew().describe()
        assert len(described) == 3
        for entry in described:
            assert entry["name"] and entry["role"] and entry["goal"]

    def test_required_agent_failure_aborts_the_run(self, populated_repository):
        crew = ReviewCrew(agents=[_FailingAgent()])
        result = crew.run("Galaxy S23")

        assert not result.success
        assert "boom" in result.error
        assert result.review is None

    def test_unknown_phone_is_reported_not_raised(self, populated_repository):
        crew = ReviewCrew(agents=[SpecificationAgent()])
        result = crew.run("Nokia 3310")
        assert not result.success
        assert result.error


@pytest.mark.llm
class TestCrewEndToEnd:
    @pytest.fixture(scope="class")
    def review(self, request):
        # Running the crew takes a couple of minutes, so it happens once for
        # the whole class.  The database is checked directly rather than through
        # the function-scoped repository fixture, which cannot be requested from
        # a class-scoped one.
        if not request.getfixturevalue("database_available"):
            pytest.skip("MySQL server is not reachable")

        from src.database.connection import session_scope
        from src.database.repository import PhoneRepository

        with session_scope() as session:
            if PhoneRepository(session).count_phones() == 0:
                pytest.skip("Database is empty - run the scraper first")

        return ReviewCrew().run("Galaxy S23")

    def test_review_is_produced(self, review):
        assert review.success
        assert review.review is not None

    def test_all_three_agents_ran(self, review):
        assert len(review.transcript) == 3
        assert all(step.success for step in review.transcript)

    def test_review_has_every_section(self, review):
        titles = [section["title"] for section in review.review["sections"]]
        assert titles == [
            "Design and Display",
            "Performance",
            "Cameras",
            "Battery and Charging",
        ]

    def test_sections_are_substantial_and_complete(self, review):
        for section in review.review["sections"]:
            assert len(section["body"].split()) > 20, section["title"]
            assert section["body"].rstrip()[-1] in ".!?\"'", section["title"]

    def test_markdown_contains_the_verified_figures(self, review):
        markdown = review.review["markdown"]
        specs = review.specifications["headline"]
        assert str(specs["battery_capacity_mah"]) in markdown
        assert review.review["phone"] in markdown

    def test_verdict_is_present(self, review):
        assert len(review.review["verdict"].split()) > 15

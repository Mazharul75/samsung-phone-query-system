"""Review Agent - writes the product review from the gathered evidence.

This is the generative specialist described in the brief.  It consumes the
Specification Agent's dossier and the Comparison Agent's positioning, then
writes the review with a LangChain chain over the local open-source model.

The review is produced section by section rather than in a single call.  A
1.5B-parameter model asked for a whole review at once drifts and starts
repeating; giving it one short, tightly-scoped brief per section keeps every
paragraph anchored to the facts it was handed.  The sections are then assembled
into the finished article.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from src.agents.base import Agent, AgentContext
from src.agents.llm_adapter import build_langchain_llm

logger = logging.getLogger(__name__)

REVIEWER_SYSTEM_PROMPT = (
    "You are a senior smartphone reviewer writing for a technology "
    "publication. You write clear, balanced prose for ordinary buyers. "
    "You only ever state figures that appear in the briefing you are given, "
    "and you never invent specifications, prices or benchmark results."
)

_SECTION_TEMPLATE = PromptTemplate.from_template(
    """You are writing one section of a product review of the {phone_name}.

Section to write: {section_title}
What this section must cover: {section_brief}

Verified facts you may use (do not use any other figures):
{facts}

Write {sentences} sentences of flowing prose for this section only.
Do not write a heading, a title, or a bullet list.
Do not mention any phone other than those named in the verified facts.
Begin writing the section now:"""
)

_VERDICT_TEMPLATE = PromptTemplate.from_template(
    """You are writing the closing verdict of a review of the {phone_name}.

Verified strengths:
{strengths}

Verified weaknesses:
{weaknesses}

Price: {price}

Write 3 sentences: state who this phone suits, name one genuine drawback, and
give a clear recommendation. Use only the facts above. Do not write a heading.
Begin writing the verdict now:"""
)


class ReviewAgent(Agent):
    """Generates a structured product review from the collected evidence."""

    name = "review_agent"
    role = "Senior product reviewer"
    goal = (
        "Combine the technical dossier and competitive analysis into a "
        "readable, accurate product review."
    )

    def __init__(self, max_new_tokens: int = 300, temperature: float = 0.4) -> None:
        self.llm = build_langchain_llm(
            system_prompt=REVIEWER_SYSTEM_PROMPT,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        self._section_chain = _SECTION_TEMPLATE | self.llm | StrOutputParser()
        self._verdict_chain = _VERDICT_TEMPLATE | self.llm | StrOutputParser()

    # ------------------------------------------------------------------
    def execute(self, context: AgentContext) -> dict[str, Any]:
        dossier = context.data.get("specifications")
        if not dossier:
            raise RuntimeError(
                "ReviewAgent requires the SpecificationAgent to run first."
            )
        analysis = context.data.get("comparison", {})

        phone_name = dossier["name"]
        headline = dossier["headline"]

        plan = _build_section_plan(dossier, analysis)
        sections: list[dict[str, str]] = []

        for section_title, section_brief, facts, sentences in plan:
            logger.info("Writing section: %s", section_title)
            body = trim_to_last_sentence(
                self._section_chain.invoke(
                    {
                        "phone_name": phone_name,
                        "section_title": section_title,
                        "section_brief": section_brief,
                        "facts": facts,
                        "sentences": sentences,
                    }
                )
            )
            sections.append({"title": section_title, "body": body})

        verdict = trim_to_last_sentence(
            self._verdict_chain.invoke(
                {
                    "phone_name": phone_name,
                    "strengths": _format_rankings(analysis.get("strengths", []))
                    or "- No standout rankings recorded.",
                    "weaknesses": _format_rankings(analysis.get("weaknesses", []))
                    or "- No significant weaknesses recorded.",
                    "price": _format_price(dossier.get("prices", [])),
                }
            )
        )

        review = {
            "phone": phone_name,
            "title": f"{phone_name} review",
            "subtitle": _build_subtitle(headline),
            "sections": sections,
            "verdict": verdict,
            "quick_specs": _quick_specs(headline),
            "markdown": _render_markdown(phone_name, headline, sections, verdict,
                                         dossier),
        }
        context.data["review"] = review
        return review

    def describe(self, output: dict[str, Any]) -> str:
        words = len(output["markdown"].split())
        return (
            f"Wrote a {len(output['sections'])}-section review of "
            f"{output['phone']} ({words} words)."
        )


def trim_to_last_sentence(text: str) -> str:
    """Drop a trailing fragment left behind by the token limit.

    Generation stops at a fixed token budget, which regularly lands mid-clause
    ("...ensures vibrant colors and").  Publishing that looks broken, so the
    text is cut back to the last completed sentence.  The fragment is only
    removed when a complete sentence precedes it, so short outputs survive
    intact.
    """
    cleaned = text.strip()
    if not cleaned:
        return cleaned

    if cleaned[-1] in ".!?\"'":
        return cleaned

    boundary = max(cleaned.rfind(mark) for mark in ".!?")
    if boundary == -1:
        return cleaned

    trimmed = cleaned[: boundary + 1].strip()
    # Only trim when a complete sentence still carries most of the text.  If
    # the unfinished fragment holds more than half the content, cutting it
    # would lose more than the ragged ending costs.
    return trimmed if len(trimmed) >= len(cleaned) / 2 else cleaned


# ---------------------------------------------------------------------------
# Briefing helpers
# ---------------------------------------------------------------------------
def _build_section_plan(
    dossier: dict[str, Any], analysis: dict[str, Any]
) -> list[tuple[str, str, str, int]]:
    """Build the per-section briefs handed to the model.

    Each entry is ``(title, brief, facts, sentence count)``.  Only facts
    relevant to a section are included, which is what keeps the model from
    wandering into unrelated specifications.
    """
    headline = dossier["headline"]
    rankings = {item["attribute"]: item for item in analysis.get("rankings", [])}

    def rank_note(attribute: str) -> str:
        item = rankings.get(attribute)
        if not item:
            return ""
        if item["is_best"]:
            return f" This is the best {attribute} of the {item['out_of']} phones compared."
        return (
            f" This ranks {item['rank']} out of {item['out_of']} phones "
            f"for {attribute}."
        )

    plan: list[tuple[str, str, str, int]] = []

    # -- Design and display -------------------------------------------
    design_facts = _facts(
        ("Released", dossier.get("announced")),
        ("Dimensions", headline.get("dimensions")),
        ("Weight", _grams(headline.get("weight_grams"))),
        ("Build", headline.get("build")),
        ("Display", _display_line(headline)),
        ("Colours", headline.get("colors")),
    )
    plan.append(
        (
            "Design and Display",
            "Describe how the phone is built and what the screen is like to use.",
            design_facts + rank_note("display size"),
            4,
        )
    )

    # -- Performance ---------------------------------------------------
    performance_facts = _facts(
        ("Chipset", headline.get("chipset")),
        ("CPU", summarise_cpu(headline.get("cpu"))),
        ("GPU", headline.get("gpu")),
        ("Memory", headline.get("memory_internal")),
        # Stated explicitly because a gap here gets filled from the model's own
        # memory: asked about a phone with no card slot, it volunteered that
        # storage could be expanded via microSD.
        (
            "Expandable storage (microSD)",
            _card_slot_note(headline.get("card_slot")),
        ),
        ("Operating system", headline.get("operating_system")),
    )
    benchmarks = dossier.get("specifications", {}).get("Our Tests", [])
    for key, value in benchmarks:
        if key.lower() == "performance":
            performance_facts += f"\n- Benchmark results: {value}"
    plan.append(
        (
            "Performance",
            "Explain how capable the processor and memory are for everyday use "
            "and gaming. Refer to the chipset and GPU by name.",
            performance_facts + rank_note("RAM"),
            4,
        )
    )

    # -- Cameras -------------------------------------------------------
    camera_facts = _facts(
        ("Main camera", headline.get("main_camera_summary")),
        ("Main sensor resolution", _mp(headline.get("main_camera_mp"))),
        ("Video recording", headline.get("main_camera_video")),
        ("Selfie camera", _mp(headline.get("selfie_camera_mp"))),
    )
    plan.append(
        (
            "Cameras",
            "Describe the camera hardware and what it can capture.",
            camera_facts + rank_note("main camera resolution"),
            4,
        )
    )

    # -- Battery -------------------------------------------------------
    battery_facts = _facts(
        ("Battery capacity", _mah(headline.get("battery_capacity_mah"))),
        ("Charging", headline.get("charging_watts")
            and f"{headline['charging_watts']}W maximum"),
        ("Measured battery test", headline.get("battery_endurance")),
    )
    plan.append(
        (
            "Battery and Charging",
            "Explain how long the phone lasts and how quickly it recharges.",
            battery_facts + rank_note("battery capacity"),
            3,
        )
    )

    return plan


def _card_slot_note(card_slot: str | None) -> str | None:
    """State the memory-card situation unambiguously.

    GSMArena records this as a bare "No", which reads as missing data rather
    than as a fact.  Spelling it out stops the model assuming the phone has
    expandable storage.
    """
    if not card_slot:
        return None
    if card_slot.strip().casefold().startswith("no"):
        return "Not supported - this phone has no microSD card slot"
    return card_slot


def summarise_cpu(cpu: str | None) -> str | None:
    """Reduce a multi-cluster CPU string to a form that cannot be misread.

    GSMArena writes the CPU as a list of clusters::

        8-core (1x3.39GHz Cortex-X4 & 3x3.1GHz Cortex-A720 &
                2x2.9GHz Cortex-A720 & 2x2.2GHz Cortex-A520)

    Asked to paraphrase that, a small model misreads the ``NxF`` multipliers
    and reports the wrong core counts - in testing it turned "1x3.39GHz" into
    "three cores at 3.39 GHz".  Instructing it not to enumerate the clusters
    did not reliably work, so the figures are pre-computed here instead and the
    raw string is never shown to the model.  The full string is still stored in
    the database and returned by the API.
    """
    if not cpu:
        return None

    # GSMArena writes the core count either as a digit ("8-core") or as a word
    # ("Octa-core"), depending on the phone's vintage.
    word_counts = {
        "dual": 2, "quad": 4, "hexa": 6, "octa": 8, "deca": 10,
    }

    cores: int | None = None
    digit_match = re.search(r"(\d+)[\s-]*core", cpu, flags=re.IGNORECASE)
    if digit_match:
        cores = int(digit_match.group(1))
    else:
        word_match = re.search(
            rf"({'|'.join(word_counts)})[\s-]*core", cpu, flags=re.IGNORECASE
        )
        if word_match:
            cores = word_counts[word_match.group(1).lower()]

    clocks = [float(value) for value in re.findall(r"([\d.]+)\s*GHz", cpu, re.I)]
    top_core = re.search(r"(Cortex-\w+|Kryo\s*\w*|Mongoose\s*\w*)", cpu, re.I)

    parts: list[str] = []
    if cores:
        parts.append(f"{cores} cores")
    if clocks:
        parts.append(f"peak clock {max(clocks):g} GHz")
    if top_core:
        parts.append(f"fastest core {top_core.group(1)}")

    return ", ".join(parts) if parts else cpu


def _facts(*pairs: tuple[str, Any]) -> str:
    lines = [f"- {label}: {value}" for label, value in pairs if value]
    return "\n".join(lines) if lines else "- No data available."


def _display_line(headline: dict[str, Any]) -> str | None:
    size = headline.get("display_size_inches")
    if not size:
        return None
    parts = [f"{size:g} inch"]
    if headline.get("display_type"):
        parts.append(headline["display_type"])
    if headline.get("display_resolution"):
        parts.append(headline["display_resolution"])
    return ", ".join(parts)


def _grams(value: Any) -> str | None:
    return f"{value:g} g" if value else None


def _mp(value: Any) -> str | None:
    return f"{value:g} MP" if value else None


def _mah(value: Any) -> str | None:
    return f"{value} mAh" if value else None


def _format_rankings(rankings: list[dict[str, Any]]) -> str:
    lines = []
    for item in rankings:
        if item["is_best"]:
            lines.append(
                f"- Best {item['attribute']} of the {item['out_of']} phones "
                f"compared ({_clean_number(item['value'])})."
            )
        else:
            lines.append(
                f"- Ranks {item['rank']} of {item['out_of']} for "
                f"{item['attribute']} ({_clean_number(item['value'])}); the "
                f"leader is {item['leader']}."
            )
    return "\n".join(lines)


def _clean_number(value: float) -> str:
    return f"{value:g}"


def _format_price(prices: list[dict[str, Any]]) -> str:
    if not prices:
        return "not listed"
    return ", ".join(f"{p['currency']} {p['amount']:,.2f}" for p in prices)


def _build_subtitle(headline: dict[str, Any]) -> str:
    bits = []
    if headline.get("display_size_inches"):
        bits.append(f"{headline['display_size_inches']:g}-inch display")
    if headline.get("chipset"):
        bits.append(headline["chipset"].split("(")[0].strip())
    if headline.get("battery_capacity_mah"):
        bits.append(f"{headline['battery_capacity_mah']} mAh battery")
    return " | ".join(bits)


def _quick_specs(headline: dict[str, Any]) -> dict[str, Any]:
    return {
        "Display": _display_line(headline),
        "Chipset": headline.get("chipset"),
        "Memory": headline.get("memory_internal"),
        "Main camera": _mp(headline.get("main_camera_mp")),
        "Selfie camera": _mp(headline.get("selfie_camera_mp")),
        "Battery": _mah(headline.get("battery_capacity_mah")),
        "Charging": headline.get("charging_watts")
        and f"{headline['charging_watts']}W",
        "Weight": _grams(headline.get("weight_grams")),
    }


def _render_markdown(
    phone_name: str,
    headline: dict[str, Any],
    sections: list[dict[str, str]],
    verdict: str,
    dossier: dict[str, Any],
) -> str:
    """Assemble the finished review as Markdown."""
    lines = [f"# {phone_name} review", ""]

    subtitle = _build_subtitle(headline)
    if subtitle:
        lines += [f"*{subtitle}*", ""]

    lines += ["## At a glance", ""]
    for label, value in _quick_specs(headline).items():
        if value:
            lines.append(f"- **{label}:** {value}")
    lines.append("")

    for section in sections:
        lines += [f"## {section['title']}", "", section["body"], ""]

    lines += ["## Verdict", "", verdict, ""]

    prices = dossier.get("prices", [])
    if prices:
        lines += [
            "## Pricing",
            "",
            f"Listed at {_format_price(prices)}.",
            "",
        ]

    lines += [
        "---",
        "",
        f"*Specifications sourced from GSMArena: {dossier.get('source_url', '')}*",
    ]
    return "\n".join(lines)

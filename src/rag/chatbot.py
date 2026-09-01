"""The conversational RAG chatbot.

Pipeline for one question
-------------------------
1. **Analyse** - classify the question and resolve any phone names against the
   database (:mod:`src.rag.query_analyzer`).
2. **Retrieve** - pull the most relevant documents from the hybrid vector
   store, restricted to the phones the question is about when they are known.
3. **Ground** - for superlatives and comparisons, add exact figures queried
   straight from SQL.  This is the step that stops the model inventing numbers.
4. **Generate** - hand the assembled context to the local open-source model
   with instructions to answer only from that context.

The answer therefore comes from retrieved evidence rather than from whatever
the base model happens to remember about Samsung phones.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.config import settings
from src.database.connection import session_scope
from src.database.repository import PhoneRepository
from src.rag.documents import Document
from src.rag.llm import LocalLLM, get_llm
from src.rag.query_analyzer import AnalyzedQuery, QueryIntent, analyze_query
from src.rag.vector_store import HybridVectorStore, get_vector_store

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a knowledgeable Samsung smartphone specialist. "
    "Answer the user's question using ONLY the reference information provided. "
    "Quote figures exactly as they appear. "
    "If the reference information does not contain the answer, say so plainly "
    "instead of guessing. Be concise, factual and helpful."
)

#: Human-readable labels and units for the columns used in rankings.
_COLUMN_LABELS: dict[str, tuple[str, str]] = {
    "battery_capacity_mah": ("Battery capacity", "mAh"),
    "charging_watts": ("Charging power", "W"),
    "display_size_inches": ("Display size", "inches"),
    "main_camera_mp": ("Main camera", "MP"),
    "selfie_camera_mp": ("Selfie camera", "MP"),
    "max_ram_gb": ("RAM", "GB"),
    "max_storage_gb": ("Storage", "GB"),
    "weight_grams": ("Weight", "g"),
    "release_year": ("Release year", ""),
    "refresh_rate_hz": ("Refresh rate", "Hz"),
}

#: Fields shown side by side when two phones are compared, with no particular
#: aspect requested.
_COMPARISON_FIELDS: tuple[tuple[str, str], ...] = (
    ("release_year", "Released"),
    ("chipset", "Chipset"),
    ("max_ram_gb", "RAM (GB)"),
    ("max_storage_gb", "Storage (GB)"),
    ("display_size_inches", "Display (in)"),
    ("refresh_rate_hz", "Refresh rate (Hz)"),
    ("main_camera_mp", "Main camera (MP)"),
    ("selfie_camera_mp", "Selfie camera (MP)"),
    ("battery_capacity_mah", "Battery (mAh)"),
    ("charging_watts", "Charging (W)"),
    ("weight_grams", "Weight (g)"),
)

#: When the question names an aspect, only those fields are compared.
#:
#: Handing a 1.5B model fifteen attributes when it was asked about one caused
#: it to drift across unrelated rows and invent figures.  Restricting the table
#: to the aspect actually asked about produced markedly more accurate answers.
_ASPECT_FIELDS: tuple[tuple[tuple[str, ...], tuple[tuple[str, str], ...]], ...] = (
    (
        ("performance", "fast", "speed", "processor", "chipset", "cpu", "gpu",
         "gaming", "benchmark", "powerful"),
        (
            ("release_year", "Released"),
            ("chipset", "Chipset"),
            ("cpu", "CPU"),
            ("gpu", "GPU"),
            ("max_ram_gb", "RAM (GB)"),
        ),
    ),
    (
        ("camera", "photo", "photography", "megapixel", "selfie", "video",
         "zoom"),
        (
            ("main_camera_mp", "Main camera (MP)"),
            ("main_camera_summary", "Main camera lenses"),
            ("selfie_camera_mp", "Selfie camera (MP)"),
            ("main_camera_video", "Video recording"),
        ),
    ),
    (
        ("battery", "charging", "endurance", "lasts", "power"),
        (
            ("battery_capacity_mah", "Battery (mAh)"),
            ("charging_watts", "Charging (W)"),
            ("battery_endurance", "Battery test result"),
        ),
    ),
    (
        ("display", "screen", "refresh", "resolution", "brightness"),
        (
            ("display_size_inches", "Display (in)"),
            ("display_type", "Display type"),
            ("display_resolution", "Resolution"),
            ("refresh_rate_hz", "Refresh rate (Hz)"),
        ),
    ),
    (
        ("design", "build", "weight", "size", "dimensions", "material"),
        (
            ("dimensions", "Dimensions"),
            ("weight_grams", "Weight (g)"),
            ("build", "Build"),
        ),
    ),
)


def _fields_for_question(question: str) -> tuple[tuple[str, str], ...]:
    """Pick the comparison fields matching the aspect named in the question."""
    text = question.lower()
    for keywords, fields in _ASPECT_FIELDS:
        if any(keyword in text for keyword in keywords):
            return fields
    return _COMPARISON_FIELDS


@dataclass
class ChatResponse:
    """A chatbot answer plus the evidence it was built from."""

    question: str
    answer: str
    intent: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    phones_referenced: list[str] = field(default_factory=list)
    context: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "intent": self.intent,
            "sources": self.sources,
            "phones_referenced": self.phones_referenced,
        }


#: Closing instruction tailored to each question type.  A small model follows
#: a specific directive far more reliably than one generic "answer the
#: question", and these directives target the exact mistakes observed during
#: testing - inventing differences between identical values, and hedging on
#: superlatives when the ranking is already given.
_INTENT_INSTRUCTIONS: dict[QueryIntent, str] = {
    QueryIntent.SPEC_LOOKUP: (
        "Answer using only the reference information above. "
        "Quote the specification figures exactly as written."
    ),
    QueryIntent.COMPARISON: (
        "Write a short comparison in full sentences (4-6 sentences) using only "
        "the verified comparison table above. Name each phone and quote its own "
        "figure. Where the table marks two values as identical, say they are "
        "the same - do NOT call an identical value higher, better or an "
        "improvement. Only describe a difference where the table shows one."
    ),
    QueryIntent.SUPERLATIVE: (
        "The ranking above is already sorted. Name the phone at position 1 as "
        "the answer and quote its figure. Do not re-order the list."
    ),
    QueryIntent.RECOMMENDATION: (
        "Recommend one phone from the reference information and justify the "
        "choice with two or three specific figures from it."
    ),
    QueryIntent.GENERAL: (
        "Answer the question using only the reference information above. "
        "If it does not contain the answer, say so."
    ),
}

#: Phrases meaning "how long does it last" rather than "how big is the cell".
_BATTERY_LIFE_TERMS = (
    "battery life",
    "lasts",
    "last longest",
    "longest battery",
    "endurance",
    "screen on time",
    "runtime",
    "battery performance",
)


def _asks_about_battery_life(question: str) -> bool:
    return any(term in question.lower() for term in _BATTERY_LIFE_TERMS)


def _format_value(phone: Any, attribute: str) -> str:
    value = getattr(phone, attribute, None)
    if value is None:
        return "not listed"
    if attribute in {"display_size_inches", "weight_grams", "main_camera_mp",
                     "selfie_camera_mp"}:
        return f"{float(value):g}"
    return str(value)


class SamsungChatbot:
    """Retrieval-augmented question answering over the Samsung phone database."""

    def __init__(
        self,
        vector_store: HybridVectorStore | None = None,
        llm: LocalLLM | None = None,
    ) -> None:
        self.vector_store = vector_store or get_vector_store()
        self.llm = llm or get_llm()

    # ------------------------------------------------------------------
    # Structured (SQL-derived) context
    # ------------------------------------------------------------------
    @staticmethod
    def _ranking_context(
        repository: PhoneRepository, analysis: AnalyzedQuery, limit: int = 6
    ) -> str:
        """A ranked table answering a superlative question."""
        column = analysis.ranking_column
        if not column:
            return ""

        # "Best battery life" is a question about measured endurance, not raw
        # capacity - and seven phones share a 5000 mAh cell, so capacity alone
        # cannot separate them.  Rank by the measured test result instead,
        # staying within one metric because the two GSMArena tests use
        # incompatible scales.
        if column == "battery_capacity_mah" and _asks_about_battery_life(
            analysis.text
        ):
            ranked = repository.rank_by_battery_life(limit=limit)
            if ranked:
                lines = [
                    "Measured battery life for the phones tested with GSMArena's "
                    "current 'active use score' (longer is better):"
                ]
                for position, phone in enumerate(ranked, start=1):
                    lines.append(
                        f"{position}. {phone.name}: {phone.battery_endurance} "
                        f"({phone.battery_capacity_mah} mAh battery)"
                    )
                return "\n".join(lines)

        try:
            ranked = repository.top_by_column(
                column, limit=limit, descending=analysis.descending
            )
        except ValueError:
            return ""

        label, unit = _COLUMN_LABELS.get(column, (column, ""))
        direction = "highest" if analysis.descending else "lowest"
        lines = [f"Ranking of all phones in the database by {label} ({direction} first):"]
        for position, phone in enumerate(ranked, start=1):
            value = _format_value(phone, column)
            extra = ""
            if column == "battery_capacity_mah" and phone.battery_endurance:
                extra = f" (measured battery test: {phone.battery_endurance})"
            lines.append(f"{position}. {phone.name}: {value} {unit}{extra}".rstrip())
        return "\n".join(lines)

    @staticmethod
    def _comparison_context(
        phones: list[Any], fields: tuple[tuple[str, str], ...] | None = None
    ) -> str:
        """A side-by-side specification table for the phones being compared.

        Each attribute is written as a separate, fully-qualified line per phone
        and followed by an explicit verdict.  A terse ``A | B`` table invited a
        small model to blur the two columns together and report one phone's
        figure for both; spelling out every value and stating which phone wins
        removes that ambiguity.
        """
        if len(phones) < 2:
            return ""

        selected = phones[:3]
        fields = fields or _COMPARISON_FIELDS
        header = " versus ".join(phone.name for phone in selected)
        lines = [f"Verified specification comparison - {header}:", ""]

        for attribute, label in fields:
            values = [_format_value(phone, attribute) for phone in selected]
            if all(value == "not listed" for value in values):
                continue

            lines.append(f"{label}:")
            for phone, value in zip(selected, values):
                lines.append(f"  - {phone.name}: {value}")

            if attribute == "battery_endurance":
                metrics = {
                    getattr(phone, "battery_endurance_metric", None)
                    for phone in selected
                }
                if len(metrics - {None}) > 1:
                    lines.append(
                        "  => These were measured with different GSMArena "
                        "battery tests and CANNOT be compared numerically."
                    )
                    lines.append("")
                    continue

            numbers = [getattr(phone, attribute, None) for phone in selected]
            if all(isinstance(n, (int, float)) for n in numbers) and (
                len(set(float(n) for n in numbers)) > 1
            ):
                best = max(selected, key=lambda p: float(getattr(p, attribute)))
                comparison = "lower" if attribute in {"weight_grams"} else "higher"
                winner = (
                    min(selected, key=lambda p: float(getattr(p, attribute)))
                    if comparison == "lower"
                    else best
                )
                lines.append(
                    f"  => {winner.name} has the {comparison} {label.lower()}."
                )
            elif len(set(values)) == 1:
                lines.append(
                    f"  => Identical - every phone listed here has {values[0]}. "
                    "This is NOT a difference."
                )
            lines.append("")

        return "\n".join(lines).strip()

    @staticmethod
    def _price_context(phones: list[Any]) -> str:
        lines: list[str] = []
        for phone in phones[:3]:
            if phone.prices:
                prices = ", ".join(
                    f"{price.currency} {float(price.amount):,.2f}"
                    for price in sorted(phone.prices, key=lambda p: p.currency)
                )
                lines.append(f"- {phone.name} listed price: {prices}")
        return ("Pricing information:\n" + "\n".join(lines)) if lines else ""

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def _retrieve(
        self, analysis: AnalyzedQuery, top_k: int
    ) -> list[tuple[Document, float]]:
        # For a comparison the retriever must not spend all its slots on one
        # phone, so an equal share is fetched per phone and then merged.
        if analysis.intent is QueryIntent.COMPARISON and analysis.is_multi_phone:
            per_phone = max(2, top_k // len(analysis.matched_phones))
            results: list[tuple[Document, float]] = []
            for phone in analysis.matched_phones:
                results.extend(
                    self.vector_store.search(
                        analysis.text, top_k=per_phone, phone_ids={phone.id}
                    )
                )
            return sorted(results, key=lambda item: item[1], reverse=True)

        return self.vector_store.search(
            analysis.text, top_k=top_k, phone_ids=analysis.phone_ids or None
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def ask(
        self,
        question: str,
        top_k: int | None = None,
        max_new_tokens: int | None = None,
    ) -> ChatResponse:
        """Answer one question and return the answer with its sources."""
        question = (question or "").strip()
        if not question:
            return ChatResponse(
                question=question,
                answer="Please ask a question about Samsung phones.",
                intent=QueryIntent.GENERAL.value,
            )

        top_k = top_k or settings.retrieval_top_k

        with session_scope() as session:
            repository = PhoneRepository(session)
            analysis = analyze_query(question, repository)
            retrieved = self._retrieve(analysis, top_k)

            structured_blocks: list[str] = []
            if analysis.intent is QueryIntent.SUPERLATIVE:
                structured_blocks.append(self._ranking_context(repository, analysis))
            elif analysis.intent is QueryIntent.COMPARISON:
                structured_blocks.append(
                    self._comparison_context(
                        analysis.matched_phones, _fields_for_question(question)
                    )
                )
            elif analysis.intent is QueryIntent.RECOMMENDATION:
                # Give the model something concrete to reason about.
                structured_blocks.append(
                    self._ranking_context(
                        repository,
                        AnalyzedQuery(
                            text=question,
                            intent=QueryIntent.SUPERLATIVE,
                            ranking_column=analysis.ranking_column
                            or "release_year",
                        ),
                        limit=5,
                    )
                )

            if analysis.matched_phones and (
                "price" in question.lower() or "cost" in question.lower()
            ):
                structured_blocks.append(self._price_context(analysis.matched_phones))

            phones_referenced = [phone.name for phone in analysis.matched_phones]

        # Assemble the reference block handed to the model.
        parts: list[str] = [block for block in structured_blocks if block]
        if retrieved:
            snippets = "\n\n".join(
                f"[{index}] {document.text}"
                for index, (document, _score) in enumerate(retrieved, start=1)
            )
            parts.append("Reference documents:\n" + snippets)

        context = "\n\n".join(parts) if parts else "No reference information found."

        prompt = (
            f"Reference information:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"{_INTENT_INSTRUCTIONS[analysis.intent]}"
        )

        answer = self.llm.generate(
            prompt,
            system_prompt=SYSTEM_PROMPT,
            max_new_tokens=max_new_tokens,
        )

        sources = [
            {
                "phone": document.phone_name,
                "section": document.section,
                "score": round(score, 4),
            }
            for document, score in retrieved
        ]

        if not phones_referenced:
            # Fall back to whichever phones the retriever surfaced.
            seen: set[str] = set()
            for document, _score in retrieved:
                if document.phone_name not in seen:
                    seen.add(document.phone_name)
                    phones_referenced.append(document.phone_name)
            phones_referenced = phones_referenced[:3]

        return ChatResponse(
            question=question,
            answer=answer,
            intent=analysis.intent.value,
            sources=sources,
            phones_referenced=phones_referenced,
            context=context,
        )


_chatbot: SamsungChatbot | None = None


def get_chatbot() -> SamsungChatbot:
    """Return the process-wide chatbot singleton."""
    global _chatbot
    if _chatbot is None:
        _chatbot = SamsungChatbot()
    return _chatbot

"""Works out what a user's question is actually asking for.

Plain vector retrieval answers "what are the camera specs of the S23?" well,
because the answer sits in one document.  It answers two other very common
question types badly:

* **Superlatives** - "which Samsung phone has the best battery life?"  The
  answer is not in any single document; it requires ranking all fifteen
  phones.  Embeddings cannot compare 5000 mAh against 3900 mAh.
* **Comparisons** - "how does the S23 compare to the S22?"  Both phones'
  documents must be retrieved, not just whichever scores highest.

So before retrieval the question is classified and any phone names in it are
resolved against the database.  The chatbot then feeds the model exact SQL
facts for superlatives and comparisons, and falls back to pure semantic search
for everything else.  This is what keeps numeric answers correct instead of
plausible-sounding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QueryIntent(str, Enum):
    """The kind of answer a question needs."""

    SPEC_LOOKUP = "spec_lookup"
    COMPARISON = "comparison"
    SUPERLATIVE = "superlative"
    RECOMMENDATION = "recommendation"
    GENERAL = "general"


#: Words that signal the user wants a ranking, mapped to the sort direction.
_SUPERLATIVE_TERMS: dict[str, bool] = {
    "best": True,
    "biggest": True,
    "largest": True,
    "highest": True,
    "most": True,
    "longest": True,
    "fastest": True,
    "top": True,
    "maximum": True,
    "smallest": False,
    "lightest": False,
    "cheapest": False,
    "lowest": False,
    "worst": False,
    "shortest": False,
}

_COMPARISON_TERMS = (
    "compare",
    "comparison",
    "versus",
    " vs ",
    " vs.",
    "difference",
    "differences",
    "better than",
    "against",
)

_RECOMMENDATION_TERMS = (
    "recommend",
    "should i buy",
    "which one should",
    "suggest",
    "worth buying",
    "good for",
)

#: Maps the vocabulary people use to the ranking column in ``phones``.
#: Order matters - the first matching phrase wins, so multi-word phrases are
#: listed before the single words they contain.
_ATTRIBUTE_COLUMNS: tuple[tuple[tuple[str, ...], str], ...] = (
    # "on a charge" and "screen on time" are listed before the charging entry
    # so that asking how long a phone lasts is read as battery life, not as
    # charging speed.
    (
        (
            "battery life",
            "battery",
            "endurance",
            "mah",
            "lasts",
            "last longest",
            "on a charge",
            "per charge",
            "screen on time",
            "runtime",
            "unplugged",
        ),
        "battery_capacity_mah",
    ),
    (("charging speed", "charging", "fast charge", "watt"), "charging_watts"),
    (("screen size", "display size", "screen", "display", "inch"), "display_size_inches"),
    (("selfie", "front camera"), "selfie_camera_mp"),
    (("camera", "megapixel", "photo"), "main_camera_mp"),
    (("ram", "memory"), "max_ram_gb"),
    (("storage", "capacity"), "max_storage_gb"),
    (("weight", "light", "heavy"), "weight_grams"),
    (("newest", "latest", "recent", "year"), "release_year"),
    (("refresh rate", "hz"), "refresh_rate_hz"),
)

#: Attributes where a *smaller* number is better.
_LOWER_IS_BETTER = {"weight_grams"}


@dataclass
class AnalyzedQuery:
    """The result of inspecting one user question."""

    text: str
    intent: QueryIntent
    phone_names: list[str] = field(default_factory=list)
    phone_ids: set[int] = field(default_factory=set)
    ranking_column: str | None = None
    descending: bool = True
    matched_phones: list[Any] = field(default_factory=list)

    @property
    def is_multi_phone(self) -> bool:
        return len(self.matched_phones) >= 2


def _detect_attribute(text: str) -> str | None:
    for phrases, column in _ATTRIBUTE_COLUMNS:
        if any(phrase in text for phrase in phrases):
            return column
    return None


def _extract_model_mentions(text: str) -> list[str]:
    """Pull Samsung model designations out of free text.

    Recognises the shapes Samsung actually uses - ``S23``, ``S23 Ultra``,
    ``Note20 Ultra``, ``Z Fold5``, ``A54`` - including an optional ``Galaxy``
    prefix and trailing trim level.
    """
    patterns = (
        r"\b(?:galaxy\s+)?(z\s*(?:fold|flip)\s*\d+)\b",
        r"\b(?:galaxy\s+)?(note\s*\d+(?:\s+ultra)?(?:\s+5g)?)\b",
        r"\b(?:galaxy\s+)?([sa]\s*\d{2})(\s+ultra|\s+plus|\s+fe|\s*\+)?(\s+5g)?\b",
    )

    mentions: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            phrase = " ".join(part for part in match.groups() if part)
            phrase = re.sub(r"\s+", " ", phrase).strip()
            if phrase and phrase.lower() not in {m.lower() for m in mentions}:
                mentions.append(phrase)
    return mentions


def analyze_query(question: str, repository: Any | None = None) -> AnalyzedQuery:
    """Classify ``question`` and resolve any phone models it names."""
    text = question.lower().strip()

    mentions = _extract_model_mentions(text)
    matched: list[Any] = []
    if repository is not None:
        seen_ids: set[int] = set()
        for mention in mentions:
            phone = repository.find_by_name(mention)
            if phone is not None and phone.id not in seen_ids:
                seen_ids.add(phone.id)
                matched.append(phone)

    ranking_column = _detect_attribute(text)

    # --- classify -----------------------------------------------------
    superlative_word = next(
        (word for word in _SUPERLATIVE_TERMS if re.search(rf"\b{word}\b", text)), None
    )
    is_comparison = any(term in text for term in _COMPARISON_TERMS) or len(matched) >= 2
    is_recommendation = any(term in text for term in _RECOMMENDATION_TERMS)

    if is_comparison and len(matched) >= 2:
        intent = QueryIntent.COMPARISON
    elif superlative_word and ranking_column:
        intent = QueryIntent.SUPERLATIVE
    elif is_recommendation:
        intent = QueryIntent.RECOMMENDATION
    elif matched:
        intent = QueryIntent.SPEC_LOOKUP
    elif is_comparison:
        intent = QueryIntent.COMPARISON
    else:
        intent = QueryIntent.GENERAL

    # --- ranking direction --------------------------------------------
    descending = True
    if superlative_word is not None:
        descending = _SUPERLATIVE_TERMS[superlative_word]
    if ranking_column in _LOWER_IS_BETTER:
        # "lightest" already means ascending; "best weight" does not.
        descending = not descending if superlative_word in {"best", "top"} else descending

    return AnalyzedQuery(
        text=question,
        intent=intent,
        phone_names=[phone.name for phone in matched] or mentions,
        phone_ids={phone.id for phone in matched},
        ranking_column=ranking_column,
        descending=descending,
        matched_phones=matched,
    )

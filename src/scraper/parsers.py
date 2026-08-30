"""Normalisation helpers that turn GSMArena's free text into typed values.

GSMArena publishes specifications as human-readable strings, for example::

    Size        6.8 inches, 114.7 cm2 (~90.2% screen-to-body ratio)
    Battery     Li-Ion 5000 mAh, non-removable
    Internal    256GB 12GB RAM, 512GB 12GB RAM, 1TB 12GB RAM

Those strings are perfect for a language model but useless for sorting and
comparing.  The functions below extract the numbers, which are then stored in
dedicated columns so questions such as "which phone has the biggest battery?"
become a single ``ORDER BY`` instead of an LLM guess.

Every parser is deliberately total: it returns ``None`` rather than raising
when a pattern is missing, because specification coverage varies by model.
"""

from __future__ import annotations

import re
import unicodedata

#: Characters GSMArena uses for layout that carry no meaning for us.
_BULLETS = "●•▪·"
_THIN_SPACES = "    "


def clean_text(value: str | None) -> str | None:
    """Collapse whitespace, drop bullet glyphs and normalise unicode.

    Returns ``None`` for values that are empty once cleaned, so that callers
    can store a real SQL ``NULL`` instead of an empty string.
    """
    if value is None:
        return None

    text = unicodedata.normalize("NFKC", value)
    for char in _THIN_SPACES:
        text = text.replace(char, " ")
    # Bullets separate alternative options ("Nano-SIM * Nano-SIM + eSIM").
    for char in _BULLETS:
        text = text.replace(char, " | ")

    text = re.sub(r"\s*\|\s*", " | ", text)
    # A bullet glyph immediately followed by a <br> yields "| |"; collapse
    # any run of separators back into a single one.
    text = re.sub(r"(?:\|\s*){2,}", "| ", text)
    text = re.sub(r"^\s*\|\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip(" |").strip()
    return text or None


def _search_float(pattern: str, text: str | None) -> float | None:
    if not text:
        return None
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _search_int(pattern: str, text: str | None) -> int | None:
    value = _search_float(pattern, text)
    return int(value) if value is not None else None


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
def parse_display_size(value: str | None) -> float | None:
    """``"6.8 inches, 114.7 cm2 (...)"`` -> ``6.8``."""
    return _search_float(r"([\d.]+)\s*inch", value)


def parse_refresh_rate(value: str | None) -> int | None:
    """``"Dynamic AMOLED 2X, 120Hz, HDR10+"`` -> ``120``."""
    return _search_int(r"(\d{2,3})\s*Hz", value)


# ---------------------------------------------------------------------------
# Body
# ---------------------------------------------------------------------------
def parse_weight(value: str | None) -> float | None:
    """``"168 g (5.93 oz)"`` -> ``168.0``."""
    return _search_float(r"([\d.]+)\s*g\b", value)


# ---------------------------------------------------------------------------
# Battery
# ---------------------------------------------------------------------------
def parse_battery_capacity(value: str | None) -> int | None:
    """``"Li-Ion 5000 mAh, non-removable"`` -> ``5000``."""
    return _search_int(r"([\d,]+)\s*mAh", value)


def parse_charging_watts(value: str | None) -> int | None:
    """Return the highest wattage mentioned in a charging description.

    A phone typically lists wired, wireless and reverse-wireless speeds; the
    headline figure users care about is the fastest one.
    """
    if not value:
        return None
    watts = [int(float(w)) for w in re.findall(r"([\d.]+)\s*W\b", value)]
    return max(watts) if watts else None


def parse_endurance(value: str | None) -> str | None:
    """Extract GSMArena's battery test result.

    Newer pages report ``Active use score 13:45h``, older ones
    ``Endurance rating 108h``.  Both are kept verbatim because they are not
    directly comparable to each other.
    """
    text = clean_text(value)
    if not text:
        return None
    match = re.search(r"(Active use score\s*[\d:]+h|Endurance rating\s*\d+h)", text, re.I)
    return match.group(1) if match else text[:120]


def parse_endurance_hours(value: str | None) -> tuple[float | None, str | None]:
    """Convert a battery test result into ``(hours, metric_name)``.

    GSMArena has used two different battery tests, and their numbers are on
    completely different scales::

        Active use score 14:49h   ->  (14.82, "active_use_score")
        Endurance rating 108h     ->  (108.0, "endurance_rating")

    The metric name is returned alongside the number precisely so that the two
    are never ranked against each other - comparing a 14-hour active-use score
    with a 108-hour endurance rating would be meaningless.
    """
    text = clean_text(value)
    if not text:
        return None, None

    active = re.search(r"Active use score\s*(\d+):(\d+)\s*h", text, flags=re.I)
    if active:
        hours, minutes = int(active.group(1)), int(active.group(2))
        return round(hours + minutes / 60, 2), "active_use_score"

    rating = re.search(r"Endurance rating\s*(\d+)\s*h", text, flags=re.I)
    if rating:
        return float(rating.group(1)), "endurance_rating"

    return None, None


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------
def parse_memory_options(value: str | None) -> tuple[int | None, int | None]:
    """Return ``(max_ram_gb, max_storage_gb)`` from an ``Internal`` string.

    Example input::

        256GB 12GB RAM, 512GB 12GB RAM, 1TB 12GB RAM

    Storage and RAM are written with the same ``GB`` unit, so they are told
    apart by the trailing ``RAM`` marker: a size followed by ``RAM`` is
    memory, anything else is storage.  ``TB`` values are converted to GB.
    """
    if not value:
        return None, None

    ram_values: list[int] = []
    storage_values: list[int] = []

    for match in re.finditer(r"([\d.]+)\s*(TB|GB|MB)\s*(RAM)?", value, flags=re.I):
        amount, unit, is_ram = match.group(1), match.group(2).upper(), match.group(3)
        try:
            size = float(amount)
        except ValueError:
            continue

        if unit == "TB":
            size *= 1024
        elif unit == "MB":
            size /= 1024

        if is_ram:
            ram_values.append(int(round(size)))
        else:
            storage_values.append(int(round(size)))

    return (
        max(ram_values) if ram_values else None,
        max(storage_values) if storage_values else None,
    )


# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------
def parse_camera_megapixels(value: str | None) -> float | None:
    """Return the resolution of the primary sensor in megapixels.

    GSMArena lists every lens in one cell, e.g.::

        200 MP, f/1.7, 24mm (wide) | 10 MP, f/2.4, 67mm (telephoto) | ...

    The main sensor is the largest figure, which is what buyers quote.
    """
    if not value:
        return None
    values = [float(m) for m in re.findall(r"([\d.]+)\s*MP", value, flags=re.I)]
    return max(values) if values else None


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
def parse_release_year(*values: str | None) -> int | None:
    """Return the first four-digit year found in the supplied strings."""
    for value in values:
        year = _search_int(r"\b(20\d{2})\b", value)
        if year:
            return year
    return None


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------
#: Symbols and codes GSMArena uses, mapped to ISO 4217 currency codes.
_CURRENCY_SYMBOLS: dict[str, str] = {
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "₹": "INR",
    "¥": "JPY",
    "₩": "KRW",
}


def parse_prices(value: str | None) -> list[tuple[str, float, str]]:
    """Parse a GSMArena price cell into ``(currency, amount, raw_text)`` tuples.

    Example input::

        $ 252.84 / EUR 299.00 / GBP 229.99 / INR 54,999

    Amounts use ``,`` as a thousands separator, which is stripped before
    conversion.  Unknown symbols are skipped rather than guessed.
    """
    text = clean_text(value)
    if not text:
        return []

    results: list[tuple[str, float, str]] = []
    seen: set[str] = set()

    symbol_class = "".join(re.escape(sym) for sym in _CURRENCY_SYMBOLS)
    pattern = rf"([{symbol_class}])\s*([\d,]+(?:\.\d+)?)"

    for match in re.finditer(pattern, text):
        symbol, amount_text = match.group(1), match.group(2)
        currency = _CURRENCY_SYMBOLS.get(symbol)
        if not currency or currency in seen:
            continue
        try:
            amount = float(amount_text.replace(",", ""))
        except ValueError:
            continue
        if amount <= 0:
            continue
        seen.add(currency)
        results.append((currency, amount, f"{symbol}{amount_text}"))

    return results


# ---------------------------------------------------------------------------
# Slugs
# ---------------------------------------------------------------------------
def slugify(name: str) -> str:
    """Turn a model name into a URL/id-safe slug (``Galaxy S23+`` -> ``galaxy-s23-plus``)."""
    text = unicodedata.normalize("NFKD", name)
    text = text.replace("+", " plus ").replace("&", " and ")
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return re.sub(r"-{2,}", "-", text)

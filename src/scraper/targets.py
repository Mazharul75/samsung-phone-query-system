"""The catalogue of Samsung models the system is built around.

Fifteen phones were chosen to give the chatbot and the review agents a data set
with genuine analytical range rather than fifteen near-identical devices:

* **Five flagship generations** (S21 -> S25) in both the base and Ultra trim,
  which makes year-over-year and tier-over-tier comparisons possible.
* **One "Fan Edition"** (S23 FE) bridging the flagship and mid-range tiers.
* **Two foldables** (Z Fold5, Z Flip5) with form factors that differ sharply
  from the slab phones.
* **Two mid-rangers** (A54, A55) so questions about value and budget options
  have real answers.

The names below are the exact labels GSMArena prints in its brand listing;
they are matched case-insensitively during URL discovery.
"""

from __future__ import annotations

#: Exact GSMArena listing names for the phones to scrape.
TARGET_MODELS: tuple[str, ...] = (
    # Flagships - base trim
    "Galaxy S21 5G",
    "Galaxy S22 5G",
    "Galaxy S23",
    "Galaxy S24",
    "Galaxy S25",
    # Flagships - Ultra trim
    "Galaxy S21 Ultra 5G",
    "Galaxy S22 Ultra 5G",
    "Galaxy S23 Ultra",
    "Galaxy S24 Ultra",
    "Galaxy S25 Ultra",
    # Fan Edition
    "Galaxy S23 FE",
    # Foldables
    "Galaxy Z Fold5",
    "Galaxy Z Flip5",
    # Mid-range
    "Galaxy A54",
    "Galaxy A55",
)


def normalise_model_name(name: str) -> str:
    """Lower-case and collapse whitespace so listing names match reliably."""
    return " ".join(name.split()).casefold()


#: Pre-computed lookup used by the scraper's discovery step.
TARGET_LOOKUP: dict[str, str] = {
    normalise_model_name(name): name for name in TARGET_MODELS
}

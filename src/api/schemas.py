"""Pydantic request and response models for the REST API.

Declaring the shapes explicitly gives FastAPI everything it needs to validate
incoming payloads and to publish an accurate OpenAPI schema, so the interactive
documentation at ``/docs`` stays correct without being written by hand.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="A question about Samsung phones.",
        examples=["What are the camera specs of the Samsung Galaxy S23?"],
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="How many documents to retrieve (defaults to the configured value).",
    )
    include_context: bool = Field(
        default=False,
        description="Return the reference text the answer was generated from.",
    )


class SourceDocument(BaseModel):
    phone: str
    section: str
    score: float


class ChatResponse(BaseModel):
    question: str
    answer: str
    intent: str = Field(
        description="How the question was classified: spec_lookup, comparison, "
        "superlative, recommendation or general."
    )
    sources: list[SourceDocument] = []
    phones_referenced: list[str] = []
    context: str | None = None
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# Phones
# ---------------------------------------------------------------------------
class PriceOut(BaseModel):
    currency: str
    amount: float


class PhoneSummary(BaseModel):
    id: int
    name: str
    slug: str
    brand: str
    release_year: int | None = None
    chipset: str | None = None
    display_size_inches: float | None = None
    main_camera_mp: float | None = None
    battery_capacity_mah: int | None = None
    max_ram_gb: int | None = None
    max_storage_gb: int | None = None
    image_url: str | None = None


class PhoneDetail(PhoneSummary):
    announced: str | None = None
    release_status: str | None = None
    dimensions: str | None = None
    weight_grams: float | None = None
    build: str | None = None
    display_type: str | None = None
    display_resolution: str | None = None
    refresh_rate_hz: int | None = None
    operating_system: str | None = None
    cpu: str | None = None
    gpu: str | None = None
    memory_internal: str | None = None
    main_camera_summary: str | None = None
    main_camera_video: str | None = None
    selfie_camera_mp: float | None = None
    battery_type: str | None = None
    charging: str | None = None
    charging_watts: int | None = None
    battery_endurance: str | None = None
    colors: str | None = None
    source_url: str
    prices: list[PriceOut] = []
    specifications: dict[str, list[dict[str, str]]] = Field(
        default_factory=dict,
        description="Every scraped specification, grouped by category.",
    )


class PhoneListResponse(BaseModel):
    total: int
    count: int
    offset: int
    phones: list[PhoneSummary]


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
class CompareRequest(BaseModel):
    phones: list[str] = Field(
        ...,
        min_length=2,
        max_length=3,
        description="Two or three model names to compare.",
        examples=[["Galaxy S24 Ultra", "Galaxy S23 Ultra"]],
    )


class ComparisonRow(BaseModel):
    attribute: str
    values: dict[str, str]
    winner: str | None = None


class CompareResponse(BaseModel):
    phones: list[str]
    rows: list[ComparisonRow]


# ---------------------------------------------------------------------------
# Reviews / agents
# ---------------------------------------------------------------------------
class ReviewRequest(BaseModel):
    phone: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="The phone to review.",
        examples=["Galaxy S24 Ultra"],
    )
    include_details: bool = Field(
        default=False,
        description="Include the raw specification dossier and competitive analysis.",
    )


class AgentStep(BaseModel):
    agent: str
    role: str
    success: bool
    summary: str = ""
    duration_seconds: float
    error: str | None = None


class ReviewSection(BaseModel):
    title: str
    body: str


class ReviewOut(BaseModel):
    phone: str
    title: str
    subtitle: str = ""
    sections: list[ReviewSection] = []
    verdict: str = ""
    quick_specs: dict[str, Any] = {}
    markdown: str


class ReviewResponse(BaseModel):
    phone: str
    success: bool
    review: ReviewOut | None = None
    agents: list[AgentStep] = []
    duration_seconds: float
    error: str | None = None
    specifications: dict[str, Any] | None = None
    comparison: dict[str, Any] | None = None


class AgentDescription(BaseModel):
    name: str
    role: str
    goal: str


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    database: bool
    vector_store: bool
    phones_indexed: int
    documents_indexed: int
    llm_backend: str
    llm_loaded: bool


class StatsResponse(BaseModel):
    phones: int
    specifications: int
    prices: int
    release_years: list[int]
    largest_battery_mah: int | None = None
    largest_display_inches: float | None = None


class ErrorResponse(BaseModel):
    detail: str

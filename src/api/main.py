"""FastAPI application exposing the Samsung phone query and review system.

Endpoints
---------
``POST /chat``            Ask the RAG chatbot a free-text question.
``POST /compare``         Side-by-side specification comparison.
``POST /reviews``         Generate a product review with the agent crew.
``GET  /phones``          Browse the catalogue.
``GET  /phones/{key}``    Full specifications for one phone (id or slug).
``GET  /phones/search``   Search the catalogue by name.
``GET  /agents``          Describe the agent crew.
``GET  /stats``           Database statistics.
``GET  /health``          Readiness of database, index and model.

Endpoints that run the language model are declared with ``def`` rather than
``async def`` on purpose: FastAPI then executes them in its worker thread pool,
so a request that takes a minute to generate a review cannot block the event
loop and stall every other caller.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Path, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from src.agents.crew import ReviewCrew, get_crew
from src.api.schemas import (
    AgentDescription,
    ChatRequest,
    ChatResponse,
    CompareRequest,
    CompareResponse,
    ComparisonRow,
    HealthResponse,
    PhoneDetail,
    PhoneListResponse,
    PhoneSummary,
    ReviewRequest,
    ReviewResponse,
    StatsResponse,
)
from src.config import settings
from src.database.connection import healthcheck, init_database, session_scope
from src.database.models import Phone
from src.database.repository import PhoneRepository
from src.rag.chatbot import SamsungChatbot, get_chatbot
from src.rag.llm import get_llm

logger = logging.getLogger(__name__)

#: Attributes shown by the /compare endpoint, with a display label and whether
#: a higher value is better (``None`` means the values are not comparable).
_COMPARE_ATTRIBUTES: tuple[tuple[str, str, bool | None], ...] = (
    ("release_year", "Released", True),
    ("chipset", "Chipset", None),
    ("cpu", "CPU", None),
    ("gpu", "GPU", None),
    ("max_ram_gb", "RAM (GB)", True),
    ("max_storage_gb", "Storage (GB)", True),
    ("display_size_inches", "Display (inches)", True),
    ("display_type", "Display type", None),
    ("refresh_rate_hz", "Refresh rate (Hz)", True),
    ("main_camera_mp", "Main camera (MP)", True),
    ("selfie_camera_mp", "Selfie camera (MP)", True),
    ("battery_capacity_mah", "Battery (mAh)", True),
    ("charging_watts", "Charging (W)", True),
    ("battery_endurance", "Battery test", None),
    ("weight_grams", "Weight (g)", False),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Prepare the database, index and model before serving traffic.

    Loading the model takes a few seconds; doing it at start-up means the
    first user request is not the one that pays for it.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.info("Starting up...")

    try:
        init_database()
    except Exception as exc:
        logger.error("Database unavailable at start-up: %s", exc)

    try:
        chatbot = get_chatbot()
        app.state.chatbot = chatbot
        logger.info("Vector store ready (%d documents).", len(chatbot.vector_store))
        chatbot.llm.warm_up()
        logger.info("Language model ready (backend=%s).", chatbot.llm.backend)
    except Exception as exc:
        logger.error("Chatbot unavailable at start-up: %s", exc)
        app.state.chatbot = None

    app.state.crew = get_crew()
    logger.info("Agent crew ready.")

    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Samsung Phone Query and Review System",
    description=(
        "A retrieval-augmented question answering and product review service "
        "built on specifications scraped from GSMArena. Runs entirely on "
        "open-source models with no external API calls."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
def get_chatbot_dependency() -> SamsungChatbot:
    chatbot = getattr(app.state, "chatbot", None)
    if chatbot is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "The chatbot is not available. Ensure the database is populated "
                "and the vector index has been built "
                "(python -m scripts.build_index)."
            ),
        )
    return chatbot


def get_crew_dependency() -> ReviewCrew:
    crew = getattr(app.state, "crew", None)
    if crew is None:
        raise HTTPException(status_code=503, detail="The agent crew is not available.")
    return crew


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------
def _number(value: Any) -> float | None:
    return None if value is None else float(value)


def _summary(phone: Phone) -> PhoneSummary:
    return PhoneSummary(
        id=phone.id,
        name=phone.name,
        slug=phone.slug,
        brand=phone.brand,
        release_year=phone.release_year,
        chipset=phone.chipset,
        display_size_inches=_number(phone.display_size_inches),
        main_camera_mp=_number(phone.main_camera_mp),
        battery_capacity_mah=phone.battery_capacity_mah,
        max_ram_gb=phone.max_ram_gb,
        max_storage_gb=phone.max_storage_gb,
        image_url=phone.image_url,
    )


def _detail(phone: Phone, repository: PhoneRepository) -> PhoneDetail:
    grouped = repository.specs_by_category(phone)
    return PhoneDetail(
        **_summary(phone).model_dump(),
        announced=phone.announced,
        release_status=phone.release_status,
        dimensions=phone.dimensions,
        weight_grams=_number(phone.weight_grams),
        build=phone.build,
        display_type=phone.display_type,
        display_resolution=phone.display_resolution,
        refresh_rate_hz=phone.refresh_rate_hz,
        operating_system=phone.operating_system,
        cpu=phone.cpu,
        gpu=phone.gpu,
        memory_internal=phone.memory_internal,
        main_camera_summary=phone.main_camera_summary,
        main_camera_video=phone.main_camera_video,
        selfie_camera_mp=_number(phone.selfie_camera_mp),
        battery_type=phone.battery_type,
        charging=phone.charging,
        charging_watts=phone.charging_watts,
        battery_endurance=phone.battery_endurance,
        colors=phone.colors,
        source_url=phone.source_url,
        prices=[
            {"currency": price.currency, "amount": float(price.amount)}
            for price in sorted(phone.prices, key=lambda p: p.currency)
        ],
        specifications={
            category: [{"key": key, "value": value} for key, value in rows]
            for category, rows in grouped.items()
        },
    )


# ---------------------------------------------------------------------------
# Service endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["service"])
def root() -> dict[str, Any]:
    """Service description and endpoint index."""
    return {
        "service": "Samsung Phone Query and Review System",
        "version": "1.0.0",
        "documentation": "/docs",
        "endpoints": {
            "POST /chat": "Ask a question about Samsung phones",
            "POST /compare": "Compare two or three phones",
            "POST /reviews": "Generate a product review (multi-agent)",
            "GET /phones": "List the catalogue",
            "GET /phones/search?q=": "Search by name",
            "GET /phones/{id_or_slug}": "Full specifications",
            "GET /agents": "Describe the agent crew",
            "GET /stats": "Database statistics",
            "GET /health": "Service health",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health() -> HealthResponse:
    """Report whether the database, index and model are ready."""
    database_ok = healthcheck()

    phones = 0
    if database_ok:
        try:
            with session_scope() as session:
                phones = PhoneRepository(session).count_phones()
        except Exception:
            database_ok = False

    chatbot = getattr(app.state, "chatbot", None)
    documents = len(chatbot.vector_store) if chatbot else 0
    llm = get_llm()

    ready = database_ok and phones > 0 and documents > 0
    return HealthResponse(
        status="ready" if ready else "degraded",
        database=database_ok,
        vector_store=documents > 0,
        phones_indexed=phones,
        documents_indexed=documents,
        llm_backend=llm.backend,
        llm_loaded=llm.is_loaded,
    )


@app.get("/stats", response_model=StatsResponse, tags=["service"])
def stats() -> StatsResponse:
    """Aggregate counts describing the scraped dataset."""
    with session_scope() as session:
        raw = PhoneRepository(session).statistics()
    return StatsResponse(
        phones=raw["phones"],
        specifications=raw["specifications"],
        prices=raw["prices"],
        release_years=raw["release_years"],
        largest_battery_mah=raw["largest_battery_mah"],
        largest_display_inches=_number(raw["largest_display_inches"]),
    )


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
@app.post("/chat", response_model=ChatResponse, tags=["chatbot"])
def chat(
    request: ChatRequest,
    chatbot: SamsungChatbot = Depends(get_chatbot_dependency),
) -> ChatResponse:
    """Answer a free-text question about Samsung phones.

    The question is classified, relevant specifications are retrieved from the
    database and vector index, and the local model answers from that evidence.
    """
    started = time.monotonic()
    try:
        result = chatbot.ask(request.question, top_k=request.top_k)
    except Exception as exc:
        logger.exception("Chat request failed")
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}") from exc

    return ChatResponse(
        question=result.question,
        answer=result.answer,
        intent=result.intent,
        sources=result.sources,
        phones_referenced=result.phones_referenced,
        context=result.context if request.include_context else None,
        elapsed_seconds=round(time.monotonic() - started, 2),
    )


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------
@app.get("/phones", response_model=PhoneListResponse, tags=["phones"])
def list_phones(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> PhoneListResponse:
    """List the phones in the catalogue, newest first."""
    with session_scope() as session:
        repository = PhoneRepository(session)
        total = repository.count_phones()
        phones = repository.list_phones(limit=limit, offset=offset)
        summaries = [_summary(phone) for phone in phones]

    return PhoneListResponse(
        total=total, count=len(summaries), offset=offset, phones=summaries
    )


@app.get("/phones/search", response_model=list[PhoneSummary], tags=["phones"])
def search_phones(
    q: str = Query(..., min_length=1, description="Part of a model name."),
    limit: int = Query(default=10, ge=1, le=50),
) -> list[PhoneSummary]:
    """Search the catalogue by model name."""
    with session_scope() as session:
        repository = PhoneRepository(session)
        return [_summary(phone) for phone in repository.search_by_name(q, limit=limit)]


@app.get("/phones/{key}", response_model=PhoneDetail, tags=["phones"])
def get_phone(
    key: str = Path(..., description="Numeric id, slug, or model name."),
) -> PhoneDetail:
    """Return the full specification sheet for one phone."""
    with session_scope() as session:
        repository = PhoneRepository(session)

        phone = None
        if key.isdigit():
            phone = repository.get_by_id(int(key))
        if phone is None:
            phone = repository.get_by_slug(key)
        if phone is None:
            phone = repository.find_by_name(key.replace("-", " "))

        if phone is None:
            raise HTTPException(status_code=404, detail=f"No phone matching {key!r}.")

        return _detail(phone, repository)


@app.get("/phones/{key}/specifications", response_class=PlainTextResponse,
         tags=["phones"])
def get_phone_specifications(key: str) -> str:
    """Return one phone's specifications as readable plain text."""
    detail = get_phone(key)
    lines = [detail.name, "=" * len(detail.name), ""]
    for category, rows in detail.specifications.items():
        lines.append(category)
        lines.append("-" * len(category))
        for row in rows:
            lines.append(f"  {row['key']}: {row['value']}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
@app.post("/compare", response_model=CompareResponse, tags=["phones"])
def compare(request: CompareRequest) -> CompareResponse:
    """Compare two or three phones attribute by attribute.

    This is a pure database operation - no language model is involved, so the
    figures are exactly what was scraped.
    """
    with session_scope() as session:
        repository = PhoneRepository(session)

        resolved: list[Phone] = []
        for name in request.phones:
            phone = repository.find_by_name(name)
            if phone is None:
                raise HTTPException(
                    status_code=404, detail=f"No phone matching {name!r}."
                )
            resolved.append(phone)

        if len({phone.id for phone in resolved}) < 2:
            raise HTTPException(
                status_code=400, detail="Please supply two different phones."
            )

        rows: list[ComparisonRow] = []
        for attribute, label, higher_is_better in _COMPARE_ATTRIBUTES:
            values: dict[str, str] = {}
            numeric: dict[str, float] = {}

            for phone in resolved:
                raw = getattr(phone, attribute, None)
                if raw is None:
                    values[phone.name] = "-"
                    continue
                if isinstance(raw, (int, float)) or hasattr(raw, "is_signed"):
                    number = float(raw)
                    numeric[phone.name] = number
                    values[phone.name] = f"{number:g}"
                else:
                    values[phone.name] = str(raw)

            if all(value == "-" for value in values.values()):
                continue

            winner = None
            if higher_is_better is not None and len(numeric) == len(resolved):
                if len(set(numeric.values())) > 1:
                    winner = (
                        max(numeric, key=numeric.get)
                        if higher_is_better
                        else min(numeric, key=numeric.get)
                    )

            rows.append(
                ComparisonRow(attribute=label, values=values, winner=winner)
            )

        return CompareResponse(
            phones=[phone.name for phone in resolved], rows=rows
        )


# ---------------------------------------------------------------------------
# Multi-agent reviews
# ---------------------------------------------------------------------------
@app.get("/agents", response_model=list[AgentDescription], tags=["agents"])
def describe_agents(
    crew: ReviewCrew = Depends(get_crew_dependency),
) -> list[AgentDescription]:
    """Describe the agents that collaborate on a review."""
    return [AgentDescription(**entry) for entry in crew.describe()]


@app.post("/reviews", response_model=ReviewResponse, tags=["agents"])
def generate_review(
    request: ReviewRequest,
    crew: ReviewCrew = Depends(get_crew_dependency),
) -> ReviewResponse:
    """Generate a full product review using the multi-agent crew.

    Three agents run in sequence: one retrieves the specifications, one ranks
    the phone against the catalogue, and one writes the review.  Generation
    runs a local model on CPU and typically takes one to two minutes.
    """
    result = crew.run(request.phone)

    if not result.success:
        raise HTTPException(
            status_code=404 if "No phone matching" in (result.error or "") else 500,
            detail=result.error or "Review generation failed.",
        )

    payload = result.to_dict(include_details=request.include_details)
    return ReviewResponse(
        phone=payload["phone"],
        success=payload["success"],
        review=payload.get("review"),
        agents=payload["agents"],
        duration_seconds=payload["duration_seconds"],
        error=payload.get("error"),
        specifications=payload.get("specifications"),
        comparison=payload.get("comparison"),
    )


@app.get("/reviews/{key}", response_class=PlainTextResponse, tags=["agents"])
def get_review_markdown(
    key: str,
    crew: ReviewCrew = Depends(get_crew_dependency),
) -> str:
    """Generate a review and return it as Markdown."""
    result = crew.run(key.replace("-", " "))
    if not result.success or not result.review:
        raise HTTPException(
            status_code=404, detail=result.error or "Review generation failed."
        )
    return result.review["markdown"]


def run() -> None:  # pragma: no cover - convenience entry point
    """Start the development server."""
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )


if __name__ == "__main__":  # pragma: no cover
    run()

# Samsung Phone Query and Review System

A system for exploring Samsung smartphones. It scrapes detailed specifications from
GSMArena into a relational database, answers natural-language questions about them with a
retrieval-augmented chatbot, generates full product reviews through a multi-agent pipeline,
and exposes everything through a REST API with a web interface.

Every model runs locally on open-source weights. There are no paid APIs, no API keys, and
no external service calls at runtime.

```
15 phones   ·   851 specifications   ·   53 price records   ·   210 indexed documents   ·   137 tests
```

---

## Contents

| | |
|---|---|
| [Overview](#overview) | What the system does |
| [Architecture](#architecture) | How the parts fit together |
| [Technology](#technology) | Libraries used and why |
| [Quick start](#quick-start) | Get it running |
| [Usage](#usage) | Running each stage |
| [Web interface](#web-interface) | The browser client |
| [API reference](#api-reference) | Endpoints |
| [Database schema](#database-schema) | Table design |
| [How it works](#how-it-works) | Design decisions |
| [Testing](#testing) | Test suite |
| [Configuration](#configuration) | Settings |
| [Project structure](#project-structure) | File layout |
| [Troubleshooting](#troubleshooting) | Common problems |

---

## Overview

The project is built in four stages, each of which runs and can be verified independently.

| Stage | Component | Description |
|:---:|---|---|
| **1** | **Scraping and storage** | Collects complete specification sheets for 15 Samsung phones from GSMArena using `requests` and BeautifulSoup, then normalises them into a MySQL schema of four related tables. Rate limited, retried, and cached on disk. |
| **2** | **RAG chatbot** | Answers questions about specifications, features, pricing and comparisons. Uses hybrid retrieval — FAISS dense vectors combined with BM25 keyword search — over 210 documents, with a locally hosted open-source language model. |
| **3** | **Multi-agent reviews** | Three LangChain-based agents collaborate to produce a full product review: one retrieves specifications, one analyses competitive position, and one writes the review. |
| **4** | **REST API** | A FastAPI service exposing chat, catalogue browsing, comparison and review generation, with automatically generated interactive documentation and a single-page web client. |

### Example questions

```
What are the camera specs of the Samsung Galaxy S23?
Which Samsung phone has the best battery life?
How does the Galaxy S23 compare to the S22 in terms of performance?
What is the screen size of the Galaxy S22?
Which phone is the lightest?
```

### Phones in the catalogue

Fifteen models were chosen to give the system genuine analytical range rather than fifteen
near-identical devices.

| Group | Models |
|---|---|
| Flagship, base | Galaxy S21 5G, S22 5G, S23, S24, S25 |
| Flagship, Ultra | Galaxy S21 Ultra 5G, S22 Ultra 5G, S23 Ultra, S24 Ultra, S25 Ultra |
| Fan Edition | Galaxy S23 FE |
| Foldable | Galaxy Z Fold5, Galaxy Z Flip5 |
| Mid-range | Galaxy A54, Galaxy A55 |

This spans five years, three price tiers and three form factors, so comparison and
recommendation questions have meaningful answers.

---

## Architecture

```
                        ┌─────────────────────────────┐
                        │         GSMArena.com        │
                        └──────────────┬──────────────┘
                                       │  requests + BeautifulSoup
                                       │  rate limited · retried · cached
        STAGE 1         ┌──────────────▼──────────────┐
                        │       GSMArenaScraper       │
                        │  discover → extract → parse │
                        └──────────────┬──────────────┘
                                       │
                        ┌──────────────▼──────────────┐
                        │   MySQL / MariaDB           │
                        │   phones · specifications   │
                        │   prices · spec_categories  │
                        └───────┬─────────────┬───────┘
                                │             │
              ┌─────────────────▼───┐   ┌─────▼──────────────────┐
        STAGE 2  Document builder   │   │   PhoneRepository      │
              │  210 chunks         │   │   exact SQL facts      │
              └─────────────────┬───┘   └─────┬──────────────────┘
                                │             │
              ┌─────────────────▼───┐         │
              │ Hybrid vector store │         │
              │   FAISS  +  BM25    │         │
              └─────────────────┬───┘         │
                                │             │
              ┌─────────────────▼─────────────▼───────────────┐
              │              SamsungChatbot                   │
              │  classify → retrieve → ground in SQL → answer │
              └─────────────────┬───────────────────────────┬─┘
                                │                           │
              ┌─────────────────▼───┐                       │
              │  Local open-source  │                       │
              │  LLM (Qwen2.5-1.5B) │◄──────────────────────┤
              └─────────────────────┘                       │
                                                            │
        STAGE 3 ┌───────────────────────────────────────┐   │
                │              ReviewCrew               │   │
                │  SpecificationAgent → ComparisonAgent │───┘
                │  → ReviewAgent   (shared context)     │
                └───────────────────┬───────────────────┘
                                    │
        STAGE 4 ┌───────────────────▼───────────────────┐
                │            FastAPI service            │
                │  /chat  /phones  /compare  /reviews   │
                └───────────────────┬───────────────────┘
                                    │  HTTP · JSON
                ┌───────────────────▼───────────────────┐
                │      Single-page web client  ( / )    │
                │  Catalogue · Chat · Compare · Review  │
                └───────────────────────────────────────┘
```

---

## Technology

| Area | Choice | Reason |
|---|---|---|
| Scraping | `requests` + `beautifulsoup4` + `lxml` | GSMArena serves static HTML, so a browser driver is unnecessary overhead. |
| Retry logic | `tenacity` | Exponential backoff on transient failures during long crawls. |
| Database | MySQL / MariaDB via `SQLAlchemy` + `PyMySQL` | Typed ORM schema; a pure-Python driver needs no compiler. |
| Configuration | `pydantic-settings` | Typed settings validated at start-up, loaded from `.env`. |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` | Apache-2.0, 90 MB, strong on short technical text. |
| Vector search | `faiss-cpu` | Exact inner-product search over normalised vectors. |
| Keyword search | `rank-bm25` | Recovers exact technical tokens that embeddings blur. |
| Generation | `transformers` — `Qwen2.5-1.5B-Instruct` | Apache-2.0, runs on CPU, follows instructions reliably. |
| Acceleration | `optimum-intel[openvino]` *(optional)* | INT8 execution of the same model; falls back to PyTorch automatically. |
| Agents | `langchain` / `langchain-core` | Prompt templates and chain composition; no API key required. |
| API | `fastapi` + `uvicorn` | Request validation and generated OpenAPI documentation. |
| Testing | `pytest` + `httpx` | 137 tests across every layer. |

### Models

Both models are open source and downloaded once from the Hugging Face Hub.

| Role | Model | Licence | Size |
|---|---|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Apache-2.0 | ~90 MB |
| Generation | `Qwen/Qwen2.5-1.5B-Instruct` | Apache-2.0 | ~3 GB |

They are cached inside the project at `.cache/` rather than in the home directory, so the
system drive is not filled unexpectedly.

---

## Quick start

### Requirements

- Python 3.10 or newer (developed on 3.13)
- MySQL 8+ or MariaDB 10.4+
- Approximately 6 GB of free disk space
- 8 GB RAM minimum

### 1. Create the environment

```bash
python -m venv .venv
```

Activate it — on Windows:

```bash
.venv\Scripts\activate
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

PyTorch is pulled from the official CPU-only wheel index, which keeps the download roughly
ten times smaller than the CUDA build.

### 3. Configure the database connection

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```ini
DB_HOST=127.0.0.1
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=samsung_phones
```

The application creates the database and its tables automatically, so no manual SQL setup
is required.

### 4. Load the data

Either rebuild from the dataset supplied with the repository, which needs no network access
and takes a few seconds:

```bash
python -m scripts.load_dataset
```

Or scrape GSMArena directly:

```bash
python -m scripts.run_scraper
```

### 5. Build the search index

```bash
python -m scripts.build_index
```

The first run downloads the models, which takes several minutes. Later runs load from disk.

### 6. Start the service

```bash
python -m scripts.run_api
```

Open **<http://127.0.0.1:8000/>** for the web interface, or
**<http://127.0.0.1:8000/docs>** for the interactive API documentation.

---

## Usage

### Stage 1 — Scraping and database

```bash
python -m scripts.run_scraper
```

The scraper walks GSMArena's Samsung brand listing to resolve each target model to its
detail page, then parses every specification table. Downloaded pages are cached under
`data/raw/html`, so re-running costs no further network requests.

| Option | Effect |
|---|---|
| `--no-cache` | Ignore cached HTML and download every page again |
| `--models "Galaxy S23" ...` | Override the built-in model list |
| `--reset` | Drop and recreate all tables before loading |
| `--delay SECONDS` | Change the delay between requests (default 2.0) |
| `-v` | Enable debug logging |

Expected output:

```
SCRAPE SUMMARY
==============================================================
{
  "requested": 15,
  "scraped": 15,
  "stored": 15,
  "total_specifications": 851,
  "database": {
    "phones": 15,
    "specifications": 851,
    "prices": 53,
    "release_years": [2021, 2022, 2023, 2024, 2025]
  }
}
```

### Stage 2 — Search index

```bash
python -m scripts.build_index
```

```
VECTOR STORE BUILT
==============================================================
Documents : 210
Phones    : 15
Location  : data/vectorstore
```

The chatbot can also be used directly from Python:

```python
from src.rag.chatbot import get_chatbot

bot = get_chatbot()
print(bot.ask("Which Samsung phone has the best battery life?").answer)
```

### Stage 3 — Multi-agent reviews

```bash
python -m scripts.generate_review "Galaxy S24 Ultra"
```

Add `--save PATH` to write the review to a Markdown file.

```
AGENT TRANSCRIPT
======================================================================
[OK ] specification_agent      0.01s  Retrieved 58 specifications across 11 categories.
[OK ] comparison_agent         0.02s  Ranked against 15 phones on 8 attributes.
[OK ] review_agent           118.00s  Wrote a 4-section review (748 words).
```

Generation runs on CPU and typically takes one to two minutes.

### Stage 4 — REST API

```bash
python -m scripts.run_api
```

Confirm the service is healthy:

```bash
curl http://127.0.0.1:8000/health
```

```json
{
  "status": "ready",
  "database": true,
  "vector_store": true,
  "phones_indexed": 15,
  "documents_indexed": 210,
  "llm_backend": "openvino",
  "llm_loaded": true
}
```

---

## Web interface

A single-page client is served at the site root, built with plain HTML and JavaScript. It
has no build step and adds no dependencies, and it consumes only the public endpoints
documented below.

| Tab | Purpose |
|---|---|
| **Dashboard** | Live dataset figures and service status |
| **Scraped Data** | Browse the catalogue; open any phone for its full specification sheet |
| **Ask a Question** | The chatbot, showing the classified question type and the retrieved sources |
| **Compare** | Side-by-side comparison with differing values highlighted |
| **Generate Review** | Runs the agent crew, showing each agent's contribution and timing |
| **API** | Endpoint reference and a link to the interactive documentation |

---

## API reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web interface |
| `GET` | `/api` | Service description and endpoint index |
| `GET` | `/health` | Readiness of database, index and model |
| `GET` | `/stats` | Dataset statistics |
| `POST` | `/chat` | Ask a natural-language question |
| `GET` | `/phones` | List the catalogue, paginated |
| `GET` | `/phones/search?q=` | Search by model name |
| `GET` | `/phones/{id_or_slug}` | Full specification sheet |
| `GET` | `/phones/{key}/specifications` | Specifications as plain text |
| `POST` | `/compare` | Compare two or three phones |
| `GET` | `/agents` | Describe the agent crew |
| `POST` | `/reviews` | Generate a review |
| `GET` | `/reviews/{key}` | Generate a review as Markdown |

### Ask a question

```bash
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "{\"question\": \"Which Samsung phone has the best battery life?\"}"
```

```json
{
  "question": "Which Samsung phone has the best battery life?",
  "answer": "Samsung Galaxy S25 Ultra has the best battery life with an active use score of 14:49h.",
  "intent": "superlative",
  "sources": [
    { "phone": "Samsung Galaxy S25 Ultra", "section": "Our Tests", "score": 0.93 }
  ],
  "phones_referenced": ["Samsung Galaxy S25 Ultra"],
  "elapsed_seconds": 10.65
}
```

### Compare phones

```bash
curl -X POST http://127.0.0.1:8000/compare -H "Content-Type: application/json" -d "{\"phones\": [\"Galaxy S24 Ultra\", \"Galaxy S23 Ultra\"]}"
```

Comparison is a pure database operation — no language model is involved, so the figures are
exactly what was scraped. Each row names a winner only where the values genuinely differ.

### Generate a review

```bash
curl -X POST http://127.0.0.1:8000/reviews -H "Content-Type: application/json" -d "{\"phone\": \"Galaxy Z Fold5\"}"
```

The response contains the finished review together with a transcript of what each agent
contributed and how long it took.

---

## Database schema

```
┌──────────────────────────────┐        ┌─────────────────────────┐
│           phones             │        │     spec_categories     │
├──────────────────────────────┤        ├─────────────────────────┤
│ id                    PK     │        │ id              PK      │
│ name, slug            UNIQUE │        │ name            UNIQUE  │
│ brand, source_url            │        │ display_order           │
│ announced, release_year      │        └───────────┬─────────────┘
│ display_size_inches          │                    │
│ display_type, refresh_rate   │                    │
│ chipset, cpu, gpu            │                    │
│ max_ram_gb, max_storage_gb   │                    │
│ main_camera_mp               │                    │
│ selfie_camera_mp             │                    │
│ battery_capacity_mah         │                    │
│ charging_watts               │                    │
│ battery_endurance_hours      │                    │
│ battery_endurance_metric     │                    │
│ weight_grams, dimensions     │                    │
└──────┬───────────────────────┘                    │
       │ 1                                          │ 1
       │                                            │
       │ N        ┌──────────────────────┐          │ N
       ├─────────►│    specifications    │◄─────────┘
       │          ├──────────────────────┤
       │          │ id            PK     │
       │          │ phone_id      FK     │
       │          │ category_id   FK     │
       │          │ spec_key             │
       │          │ spec_value           │
       │          │ position             │
       │          └──────────────────────┘
       │ N        ┌──────────────────────┐
       └─────────►│        prices        │
                  ├──────────────────────┤
                  │ id            PK     │
                  │ phone_id      FK     │
                  │ currency, amount     │
                  │ raw_text             │
                  └──────────────────────┘
```

The design stores the scraped data in two complementary forms:

- **`specifications`** keeps every published row verbatim, so nothing collected is ever lost.
- **`phones`** keeps the *parsed* headline values in typed numeric columns.

That second form is what turns a question such as "which phone has the biggest battery?"
into a single `ORDER BY` rather than something the language model has to infer.

---

## How it works

### Retrieval is hybrid, and grounded in SQL

Phone questions are full of exact tokens — `Snapdragon 8 Gen 2`, `120Hz`, `S23 Ultra` —
where pure embedding search is unreliable. Retrieval therefore combines **FAISS dense
vectors** for meaning with **BM25** for exact terms, fusing their normalised scores.

Retrieval alone still answers two common question types poorly, so every question is
classified before it is answered:

| Intent | How it is answered |
|---|---|
| `spec_lookup` | Retrieval restricted to the phone named in the question |
| `comparison` | A verified side-by-side table built from SQL, narrowed to the aspect asked about |
| `superlative` | A ranked table produced by SQL — embeddings cannot compare 5000 mAh against 3900 mAh |
| `recommendation` | A ranking combined with retrieved context |
| `general` | Plain hybrid retrieval |

The language model never produces a ranking; it reports one that has already been computed.

### Battery life is measured, not inferred

Seven of the fifteen phones share a 5000 mAh cell, so capacity alone cannot answer "which
has the best battery life". GSMArena's measured test result is parsed into a numeric column
instead — but that test has used **two incompatible scales** (*active use score* in hours,
and an older *endurance rating*). Both the value and the metric that produced it are stored,
and rankings never mix the two.

### Reviews are composed section by section

A 1.5B-parameter model asked to write an entire review in one call drifts and repeats
itself. The review agent instead issues one tightly-scoped brief per section, containing
only the facts relevant to that section, then assembles the results. Any trailing fragment
left by the token limit is trimmed back to the last complete sentence.

Values that a small model reliably misreads are pre-computed rather than passed raw. A CPU
string such as `8-core (1x3.39GHz Cortex-X4 & 3x3.1GHz Cortex-A720 & …)` is reduced to
`8 cores, peak clock 3.39 GHz` before the model sees it.

### Data fidelity

GSMArena separates individual camera lenses inside a single table cell with `<br>` tags.
Collapsing that cell to plain text merges three lenses into one string, and every downstream
consumer then treats them as a single sensor. The scraper converts those breaks into an
explicit separator before extracting text, which keeps each lens distinct in the database,
in the search index and in generated answers.

### Inference backend

The generation model is served through one of two interchangeable backends, selected
automatically at start-up:

- **`openvino`** — the same model executed with INT8 weights. On the reference machine this
  raised generation from 2 to roughly 10 tokens per second with no loss of answer quality.
  Conversion happens once and is cached.
- **`transformers`** — plain PyTorch on CPU. Slower, but depends only on packages that are
  required anyway.

OpenVINO is an optional *runtime* for the same model, not a different model. Removing
`optimum-intel[openvino]` from `requirements.txt` makes the application fall back to PyTorch
with no code changes.

---

## Testing

```bash
pytest
```

**137 tests** cover specification parsing, the database layer, retrieval, the agents, every
API endpoint and the web client.

| Command | Effect |
|---|---|
| `pytest` | Run everything (about 5 minutes) |
| `pytest -m "not llm"` | Skip the tests that load the language model (about 11 seconds) |
| `pytest tests/test_scraper.py` | Run one module |

Parsing tests run against committed HTML fixtures, so they need neither network access nor a
database. Tests that require MySQL skip automatically when no server is reachable, which
means a fresh checkout still runs the full parser suite successfully.

---

## Configuration

All settings are read from environment variables, optionally via a `.env` file.

| Variable | Default | Purpose |
|---|---|---|
| `DB_HOST` | `127.0.0.1` | Database host |
| `DB_PORT` | `3306` | Database port |
| `DB_USER` | `root` | Database user |
| `DB_PASSWORD` | — | Database password |
| `DB_NAME` | `samsung_phones` | Database name |
| `SCRAPER_DELAY_SECONDS` | `2.0` | Delay between requests |
| `SCRAPER_USE_CACHE` | `true` | Reuse downloaded HTML |
| `SCRAPER_MAX_LISTING_PAGES` | `15` | Listing pages to search when resolving models |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `LLM_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | Language model |
| `LLM_MAX_NEW_TOKENS` | `512` | Generation limit |
| `LLM_BACKEND` | auto | `openvino` or `transformers` |
| `RETRIEVAL_TOP_K` | `6` | Documents retrieved per query |
| `API_PORT` | `8000` | API port |

---

## Project structure

```
.
├── data/
│   └── samsung_phones_dataset.json   Scraped dataset, version controlled
├── scripts/
│   ├── run_scraper.py                Stage 1 — scrape and store
│   ├── load_dataset.py               Stage 1 — rebuild the database offline
│   ├── build_index.py                Stage 2 — build the search index
│   ├── generate_review.py            Stage 3 — run the agent crew
│   └── run_api.py                    Stage 4 — start the API
├── src/
│   ├── config.py                     Central settings
│   ├── database/
│   │   ├── models.py                 SQLAlchemy ORM models
│   │   ├── connection.py             Engine, sessions, schema bootstrap
│   │   └── repository.py             All database queries
│   ├── scraper/
│   │   ├── http_client.py            Rate limiting, retries, caching
│   │   ├── parsers.py                Text to typed values
│   │   ├── gsmarena.py               Discovery and extraction
│   │   ├── targets.py                The fifteen target models
│   │   └── pipeline.py               Scrape to database
│   ├── rag/
│   │   ├── documents.py              Database rows to search documents
│   │   ├── embeddings.py             Sentence-embedding wrapper
│   │   ├── vector_store.py           FAISS and BM25 hybrid retrieval
│   │   ├── query_analyzer.py         Intent and model-name detection
│   │   ├── llm.py                    Local model with pluggable backend
│   │   └── chatbot.py                The RAG pipeline
│   ├── agents/
│   │   ├── base.py                   Agent contract and shared context
│   │   ├── llm_adapter.py            LangChain adapter
│   │   ├── specification_agent.py    Retrieves specifications
│   │   ├── comparison_agent.py       Competitive positioning
│   │   ├── review_agent.py           Writes the review
│   │   └── crew.py                   Orchestration
│   └── api/
│       ├── schemas.py                Request and response models
│       ├── main.py                   FastAPI application
│       └── static/index.html         Web client
├── tests/                            137 tests
├── requirements.txt
└── .env.example
```

---

## Troubleshooting

**`Can't connect to MySQL server`**
Confirm the server is running and that the credentials in `.env` are correct. Verify with
`mysql -u root -p -e "SELECT 1"`.

**`The chatbot is not available` (HTTP 503)**
The search index has not been built:

```bash
python -m scripts.build_index
```

**`Database is empty`**
Populate it from the supplied dataset:

```bash
python -m scripts.load_dataset
```

**`No space left on device` while downloading models**
The models require approximately 5 GB. They are cached in the project's `.cache/` directory,
so the drive holding the project needs the free space.

**Generation is slow**
Check that the fast backend is active — `/health` should report `"llm_backend": "openvino"`.
If it reports `transformers`, reinstall the optional accelerator:

```bash
pip install "optimum-intel[openvino]"
```

**The scraper returns no phones**
GSMArena may have changed its page structure, or the request was blocked. Re-run with
`--no-cache` and `-v` to inspect the failing request.

**Schema errors after changing the models**
`create_all` only creates missing tables; it never alters an existing one. Rebuild the
schema with:

```bash
python -m scripts.run_scraper --reset
```

---

## Data source

Specifications are scraped from [GSMArena](https://www.gsmarena.com/). The scraper requests
only pages permitted by the site's `robots.txt`, identifies itself with a standard browser
user agent, waits two seconds between requests, and caches every response so repeated runs
generate no additional traffic. The data is used here for educational and demonstration
purposes.

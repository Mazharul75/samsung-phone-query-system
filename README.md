# Samsung Phone Query and Review System

An intelligent system for exploring Samsung smartphones. It scrapes detailed
specifications from GSMArena into a relational database, answers free-text
questions about them with a retrieval-augmented chatbot, generates full product
reviews through a multi-agent pipeline, and exposes everything over a REST API.

Every model runs **locally on open-source weights** — there are no paid APIs, no
API keys and no external service calls at runtime.

---

## Contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the project](#running-the-project)
  - [Phase 1 — Scraping and database](#phase-1--scraping-and-database)
  - [Phase 2 — RAG chatbot](#phase-2--rag-chatbot)
  - [Phase 3 — Multi-agent reviews](#phase-3--multi-agent-reviews)
  - [Phase 4 — REST API](#phase-4--rest-api)
- [API reference](#api-reference)
- [Database schema](#database-schema)
- [Design notes](#design-notes)
- [Testing](#testing)
- [Configuration](#configuration)
- [Project structure](#project-structure)
- [Troubleshooting](#troubleshooting)

---

## What it does

| Capability | Description |
|---|---|
| **Web scraping** | Collects complete specification sheets for 15 Samsung phones from GSMArena using `requests` + BeautifulSoup, with rate limiting, retries and an on-disk HTML cache. |
| **Structured storage** | Normalises the data into a MySQL/MariaDB schema of four related tables — 15 phones, 851 specification rows and 52 price points. |
| **RAG chatbot** | Answers questions about specifications, features, pricing and comparisons using hybrid retrieval (FAISS dense vectors + BM25) over 210 documents, with a locally hosted open-source LLM. |
| **Multi-agent system** | Three LangChain-based agents collaborate to produce a full product review: one retrieves specifications, one analyses competitive position, one writes the review. |
| **REST API** | A FastAPI service exposing chat, catalogue browsing, comparison and review generation, with interactive documentation at `/docs`. |

### Example questions the chatbot handles

```
What are the camera specs of the Samsung Galaxy S23?
Which Samsung phone has the best battery life?
How does the Galaxy S23 compare to the S22 in terms of performance?
What is the screen size of the Galaxy S22?
Which phone is the lightest?
```

---

## Architecture

```
                   ┌──────────────────────────────────────────┐
                   │              GSMArena.com                │
                   └──────────────────┬───────────────────────┘
                                      │  requests + BeautifulSoup
                                      │  (rate limited, cached, retried)
                   ┌──────────────────▼───────────────────────┐
   PHASE 1         │           GSMArenaScraper                │
                   │   discovery → extraction → normalisation │
                   └──────────────────┬───────────────────────┘
                                      │
                   ┌──────────────────▼───────────────────────┐
                   │      MySQL / MariaDB  (SQLAlchemy)       │
                   │  phones · specifications · prices ·      │
                   │  spec_categories                         │
                   └───────┬──────────────────────┬───────────┘
                           │                      │
             ┌─────────────▼──────────┐   ┌───────▼─────────────────┐
   PHASE 2   │   Document builder     │   │   PhoneRepository       │
             │   210 chunks           │   │   (exact SQL facts)     │
             └─────────────┬──────────┘   └───────┬─────────────────┘
                           │                      │
             ┌─────────────▼──────────┐           │
             │  Hybrid vector store   │           │
             │  FAISS  +  BM25        │           │
             └─────────────┬──────────┘           │
                           │                      │
             ┌─────────────▼──────────────────────▼─────────────┐
             │                SamsungChatbot                    │
             │  analyse → retrieve → ground in SQL → generate   │
             └─────────────┬───────────────────────────────────┬┘
                           │                                   │
             ┌─────────────▼──────────┐                        │
             │   Local open-source    │                        │
             │   LLM (Qwen2.5-1.5B)   │◄───────────────────────┤
             └────────────────────────┘                        │
                                                               │
   PHASE 3   ┌───────────────────────────────────────────────┐ │
             │                 ReviewCrew                    │ │
             │  SpecificationAgent → ComparisonAgent →       │─┘
             │  ReviewAgent  (shared AgentContext)           │
             └───────────────────────┬───────────────────────┘
                                     │
   PHASE 4   ┌───────────────────────▼───────────────────────┐
             │                FastAPI service                │
             │  /chat  /phones  /compare  /reviews  /agents  │
             └───────────────────────────────────────────────┘
```

---

## Requirements

- **Python 3.10 or newer** (developed on 3.13)
- **MySQL 8+ or MariaDB 10.4+** running locally
- **~6 GB free disk space** for the Python environment and model weights
- **8 GB RAM minimum** (the language model uses roughly 1.5 GB)
- An internet connection for the first run only — to install packages, download
  the models, and scrape GSMArena

---

## Installation

### 1. Clone and enter the project

```bash
git clone <your-repository-url>
cd samsung-phone-query-system
```

### 2. Create a virtual environment

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This installs PyTorch from the official CPU-only wheel index, which keeps the
download roughly ten times smaller than the CUDA build.

### 4. Create the database

Connect to your MySQL/MariaDB server and create an empty database:

```bash
mysql -u root -p -e "CREATE DATABASE samsung_phones CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
```

The application also creates the database automatically if the configured user
has permission, so this step is optional.

### 5. Configure the connection

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

Then edit `.env`:

```ini
DB_HOST=127.0.0.1
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=samsung_phones
```

---

## Running the project

Each phase runs independently and can be verified on its own.

### Phase 1 — Scraping and database

Create the schema, scrape GSMArena and populate the database:

```bash
python -m scripts.run_scraper
```

The scraper walks GSMArena's Samsung brand listing to resolve each target model
to its detail page, then parses every specification table. Pages are cached
under `data/raw/html`, so re-running costs no further network requests.

Useful options:

```bash
python -m scripts.run_scraper --no-cache
```

```bash
python -m scripts.run_scraper --models "Galaxy S23" "Galaxy S24 Ultra"
```

```bash
python -m scripts.run_scraper --reset
```

**Expected output**

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
    "prices": 52,
    "release_years": [2021, 2022, 2023, 2024, 2025]
  }
}
```

#### Loading without scraping

The scraped dataset ships with the repository, so the database can be rebuilt
offline in a couple of seconds:

```bash
python -m scripts.load_dataset
```

### Phase 2 — RAG chatbot

Build the search index from the database:

```bash
python -m scripts.build_index
```

**Expected output**

```
VECTOR STORE BUILT
==============================================================
Documents : 210
Phones    : 15
Location  : data/vectorstore
```

The first run downloads the embedding model (~90 MB) and the language model
(~3 GB) from the Hugging Face Hub into `.cache/`. Subsequent runs load from
disk.

Ask questions from the API (see Phase 4), or directly in Python:

```python
from src.rag.chatbot import get_chatbot

bot = get_chatbot()
print(bot.ask("Which Samsung phone has the best battery life?").answer)
```

### Phase 3 — Multi-agent reviews

Generate a complete product review:

```bash
python -m scripts.generate_review "Galaxy S24 Ultra"
```

Save it to a file:

```bash
python -m scripts.generate_review "Galaxy S23 Ultra" --save reviews/s23-ultra.md
```

**Expected output**

```
AGENT TRANSCRIPT
======================================================================
[OK ] specification_agent      0.01s  Retrieved 58 specifications for Samsung Galaxy S24 Ultra across 11 categories.
[OK ] comparison_agent         0.02s  Ranked against 15 phones on 8 attributes; found 5 standout strengths and 3 close rivals.
[OK ] review_agent           118.00s  Wrote a 4-section review of Samsung Galaxy S24 Ultra (748 words).
```

followed by the finished review in Markdown. Generation runs on CPU and takes
roughly one to two minutes.

### Phase 4 — REST API

Start the server:

```bash
python -m scripts.run_api
```

Then open **<http://127.0.0.1:8000/docs>** for interactive documentation.

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

## API reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Service description and endpoint index |
| `GET` | `/health` | Readiness of database, index and model |
| `GET` | `/stats` | Dataset statistics |
| `POST` | `/chat` | Ask a free-text question |
| `GET` | `/phones` | List the catalogue (paginated) |
| `GET` | `/phones/search?q=` | Search by model name |
| `GET` | `/phones/{id_or_slug}` | Full specification sheet |
| `GET` | `/phones/{key}/specifications` | Specifications as plain text |
| `POST` | `/compare` | Compare two or three phones |
| `GET` | `/agents` | Describe the agent crew |
| `POST` | `/reviews` | Generate a review (multi-agent) |
| `GET` | `/reviews/{key}` | Generate a review as Markdown |

### Ask a question

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Which Samsung phone has the best battery life?"}'
```

```json
{
  "question": "Which Samsung phone has the best battery life?",
  "answer": "Samsung Galaxy S25 Ultra has the best battery life with an active use score of 14:49h.",
  "intent": "superlative",
  "sources": [
    { "phone": "Samsung Galaxy A55", "section": "Our Tests", "score": 0.93 }
  ],
  "phones_referenced": ["Samsung Galaxy S25 Ultra"],
  "elapsed_seconds": 10.65
}
```

### Compare phones

```bash
curl -X POST http://127.0.0.1:8000/compare \
  -H "Content-Type: application/json" \
  -d '{"phones": ["Galaxy S24 Ultra", "Galaxy S23 Ultra"]}'
```

Comparison is a pure database operation — no language model is involved, so the
figures are exactly what was scraped, and each row names the winner where the
values genuinely differ.

### Generate a review

```bash
curl -X POST http://127.0.0.1:8000/reviews \
  -H "Content-Type: application/json" \
  -d '{"phone": "Galaxy Z Fold5"}'
```

The response contains the finished review plus a transcript of what each agent
contributed and how long it took.

---

## Database schema

```
┌────────────────────────────┐          ┌──────────────────────────┐
│          phones            │          │     spec_categories      │
├────────────────────────────┤          ├──────────────────────────┤
│ id              PK         │          │ id            PK         │
│ name, slug      UNIQUE     │          │ name          UNIQUE     │
│ brand, source_url          │          │ display_order            │
│ release_year, announced    │          └────────────┬─────────────┘
│ display_size_inches        │                       │
│ display_type, refresh_rate │                       │
│ chipset, cpu, gpu          │                       │
│ max_ram_gb, max_storage_gb │                       │
│ main_camera_mp             │                       │
│ selfie_camera_mp           │                       │
│ battery_capacity_mah       │                       │
│ charging_watts             │                       │
│ battery_endurance_hours    │                       │
│ battery_endurance_metric   │                       │
│ weight_grams, dimensions   │                       │
└─────────┬──────────────────┘                       │
          │ 1                                        │ 1
          │                                          │
          │ N          ┌────────────────────────┐    │ N
          ├───────────►│     specifications     │◄───┘
          │            ├────────────────────────┤
          │            │ id            PK       │
          │            │ phone_id      FK       │
          │            │ category_id   FK       │
          │            │ spec_key               │
          │            │ spec_value             │
          │            │ position               │
          │            └────────────────────────┘
          │ N          ┌────────────────────────┐
          └───────────►│         prices         │
                       ├────────────────────────┤
                       │ id            PK       │
                       │ phone_id      FK       │
                       │ currency, amount       │
                       │ raw_text, captured_at  │
                       └────────────────────────┘
```

The design stores the data twice, deliberately:

- **`specifications`** keeps every published row verbatim, so nothing scraped is
  ever lost.
- **`phones`** holds the *parsed* headline values in typed numeric columns.

That second copy is what makes questions like "which phone has the biggest
battery?" a single `ORDER BY` instead of something the language model has to
guess at.

---

## Design notes

### Retrieval is hybrid, and grounded in SQL

Phone questions are full of exact tokens — `Snapdragon 8 Gen 2`, `120Hz`,
`S23 Ultra` — where pure embedding search is unreliable. Retrieval therefore
fuses **FAISS dense vectors** (meaning) with **BM25** (exact terms).

Retrieval alone still cannot answer two common question types, so incoming
questions are classified first:

| Intent | How it is answered |
|---|---|
| `spec_lookup` | Retrieval restricted to the phone named in the question |
| `comparison` | A verified side-by-side table built from SQL, narrowed to the aspect asked about |
| `superlative` | A ranked table produced by SQL — embeddings cannot compare 5000 mAh against 3900 mAh |
| `recommendation` | A ranking plus retrieved context |
| `general` | Plain hybrid retrieval |

The language model never invents a ranking; it reports one that has already been
computed.

### Battery life is measured, not assumed

Seven of the fifteen phones share a 5000 mAh cell, so capacity alone cannot
answer "which has the best battery life". GSMArena's measured test result is
parsed into a numeric column — but it has used **two incompatible scales**
(*active use score* in hours, and the older *endurance rating*). Both the metric
and the value are stored, and rankings never mix the two.

### Reviews are written section by section

A 1.5B-parameter model asked for an entire review in one call drifts and starts
repeating itself. The Review Agent instead issues one tightly-scoped brief per
section, containing only the facts relevant to that section, and assembles the
results. Any trailing fragment left by the token limit is trimmed back to the
last complete sentence.

### Data fidelity

GSMArena separates individual camera lenses inside a single table cell with
`<br>` tags. Collapsing that cell to plain text merges three lenses into one
run-on string, and every downstream consumer then treats them as a single
sensor. The scraper converts those breaks into an explicit separator before
extracting text, which keeps each lens distinct in the database, in the search
index and in generated answers.

---

## Testing

```bash
pytest
```

**127 tests** cover parsing, the database layer, retrieval, the agents and every
API endpoint.

Parsing tests run against committed HTML fixtures, so they need neither network
access nor a database. Tests that require MySQL skip automatically when no
server is reachable.

Skip the slower tests that invoke the language model:

```bash
pytest -m "not llm"
```

Run one phase's tests:

```bash
pytest tests/test_scraper.py
```

---

## Configuration

All settings are read from environment variables or `.env`.

| Variable | Default | Purpose |
|---|---|---|
| `DB_HOST` | `127.0.0.1` | Database host |
| `DB_PORT` | `3306` | Database port |
| `DB_USER` | `root` | Database user |
| `DB_PASSWORD` | — | Database password |
| `DB_NAME` | `samsung_phones` | Database name |
| `SCRAPER_DELAY_SECONDS` | `2.0` | Delay between requests |
| `SCRAPER_USE_CACHE` | `true` | Reuse downloaded HTML |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `LLM_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | Language model |
| `LLM_MAX_NEW_TOKENS` | `512` | Generation limit |
| `LLM_BACKEND` | auto | `openvino` or `transformers` |
| `RETRIEVAL_TOP_K` | `6` | Documents retrieved per query |
| `API_PORT` | `8000` | API port |

### Models

Both models are open source and downloaded once from the Hugging Face Hub:

| Role | Model | Licence | Size |
|---|---|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Apache-2.0 | ~90 MB |
| Generation | `Qwen/Qwen2.5-1.5B-Instruct` | Apache-2.0 | ~3 GB |

They are cached inside the project at `.cache/` rather than the home directory,
so the system drive is not silently filled.

### Inference backend

The generation model is served through one of two interchangeable backends,
selected automatically:

- **`openvino`** — the same model executed with INT8 weights. On the reference
  machine this raised generation from 2 to roughly 10 tokens per second with no
  loss of answer quality. Conversion happens once and is cached.
- **`transformers`** — plain PyTorch on CPU. Slower, but depends only on
  packages that are required anyway.

OpenVINO is an optional *runtime* for the same Hugging Face model, not a
different model. Removing `optimum-intel[openvino]` from `requirements.txt`
makes the application fall back to PyTorch with no code changes.

---

## Project structure

```
.
├── data/
│   └── samsung_phones_dataset.json   Scraped dataset (ships with the repo)
├── scripts/
│   ├── run_scraper.py                Phase 1 — scrape and store
│   ├── load_dataset.py               Phase 1 — rebuild the DB offline
│   ├── build_index.py                Phase 2 — build the search index
│   ├── generate_review.py            Phase 3 — run the agent crew
│   └── run_api.py                    Phase 4 — start the API
├── src/
│   ├── config.py                     Central settings
│   ├── database/
│   │   ├── models.py                 SQLAlchemy ORM models
│   │   ├── connection.py             Engine, sessions, schema bootstrap
│   │   └── repository.py             All queries live here
│   ├── scraper/
│   │   ├── http_client.py            Rate limiting, retries, caching
│   │   ├── parsers.py                Text → typed values
│   │   ├── gsmarena.py               Discovery and extraction
│   │   ├── targets.py                The 15 phone models
│   │   └── pipeline.py               Scrape → database
│   ├── rag/
│   │   ├── documents.py              Database rows → search documents
│   │   ├── embeddings.py             Sentence-embedding wrapper
│   │   ├── vector_store.py           FAISS + BM25 hybrid retrieval
│   │   ├── query_analyzer.py         Intent and model-name detection
│   │   ├── llm.py                    Local model with pluggable backend
│   │   └── chatbot.py                The RAG pipeline
│   ├── agents/
│   │   ├── base.py                   Agent contract and shared context
│   │   ├── llm_adapter.py            LangChain LLM adapter
│   │   ├── specification_agent.py    Retrieves specifications
│   │   ├── comparison_agent.py       Competitive positioning
│   │   ├── review_agent.py           Writes the review
│   │   └── crew.py                   Orchestration
│   └── api/
│       ├── schemas.py                Request/response models
│       └── main.py                   FastAPI application
├── tests/                            127 tests
├── requirements.txt
└── .env.example
```

---

## Troubleshooting

**`Can't connect to MySQL server`**
Check that the server is running and that the credentials in `.env` are
correct. Verify with `mysql -u root -p -e "SELECT 1"`.

**`The chatbot is not available` (HTTP 503)**
The vector index has not been built. Run:

```bash
python -m scripts.build_index
```

**`Database is empty`**
Populate it first, either by scraping or from the shipped dataset:

```bash
python -m scripts.load_dataset
```

**`No space left on device` while downloading models**
The models need roughly 5 GB. They are cached in the project's `.cache/`
directory, so ensure the drive holding the project has room.

**Generation is slow**
Confirm the fast backend is active — `/health` should report
`"llm_backend": "openvino"`. If it reports `transformers`, reinstall the
optional accelerator:

```bash
pip install "optimum-intel[openvino]"
```

**Scraper returns no phones**
GSMArena may have changed its page structure, or the request was blocked.
Re-run with `--no-cache` and `-v` to see the failing request.

---

## Data source

Specifications are scraped from [GSMArena](https://www.gsmarena.com/). The
scraper requests only pages permitted by the site's `robots.txt`, identifies
itself with a standard browser user agent, waits two seconds between requests
and caches every response so repeated runs generate no additional traffic. The
data is used here for educational and demonstration purposes.

# Samsung Phone Query and Review System

Scrapes Samsung phone specifications from GSMArena into a MySQL database, answers questions
about them with a retrieval-augmented chatbot, generates product reviews through a
multi-agent pipeline, and serves it all over a REST API with a web interface.

All models are open source and run locally — no paid APIs, no API keys.

`15 phones` · `851 specifications` · `53 prices` · `210 indexed documents` · `137 tests`

---

## What it does

| Stage | Component | Description |
|:-:|---|---|
| 1 | **Scraping & database** | BeautifulSoup scraper collecting 15 Samsung phones into four normalised MySQL tables. Rate limited, retried, cached. |
| 2 | **RAG chatbot** | Hybrid retrieval (FAISS + BM25) over 210 documents, answered by a local open-source model grounded in SQL facts. |
| 3 | **Multi-agent reviews** | Three LangChain agents: one retrieves specifications, one ranks the phone against the catalogue, one writes the review. |
| 4 | **REST API** | FastAPI service with generated OpenAPI docs and a single-page web client. |

**Example questions**

```
What are the camera specs of the Samsung Galaxy S23?
Which Samsung phone has the best battery life?
How does the Galaxy S23 compare to the S22 in terms of performance?
```

**Catalogue** — Galaxy S21 to S25 (base and Ultra), S23 FE, Z Fold5, Z Flip5, A54, A55.
Five years, three price tiers, three form factors, so comparisons have real answers.

---

## Technology

| Layer | Stack |
|---|---|
| Scraping | `requests` · `beautifulsoup4` · `lxml` · `tenacity` |
| Database | MySQL / MariaDB · `SQLAlchemy` · `PyMySQL` |
| Retrieval | `sentence-transformers` (all-MiniLM-L6-v2) · `faiss-cpu` · `rank-bm25` |
| Generation | `transformers` · Qwen2.5-1.5B-Instruct (Apache-2.0) · OpenVINO INT8 *(optional)* |
| Agents | `langchain` |
| API | `fastapi` · `uvicorn` |
| Tests | `pytest` |

---

## Setup

Requires Python 3.10+, MySQL 8+ or MariaDB 10.4+, and ~6 GB free disk space.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set your database credentials. The application creates the
database and tables automatically.

```bash
python -m scripts.load_dataset    # load the dataset shipped with the repo
python -m scripts.build_index     # build the search index
python -m scripts.run_api         # start the service
```

Open **<http://127.0.0.1:8000/>** for the web interface, or **`/docs`** for the API
documentation.

> The first index build downloads the models (~3 GB), cached in `.cache/`.

---

## Commands

| Command | Purpose |
|---|---|
| `python -m scripts.run_scraper` | Scrape GSMArena and populate the database |
| `python -m scripts.load_dataset` | Rebuild the database offline from the shipped dataset |
| `python -m scripts.build_index` | Build the FAISS + BM25 search index |
| `python -m scripts.generate_review "Galaxy S24 Ultra"` | Run the agent crew |
| `python -m scripts.run_api` | Start the API server |
| `pytest` | Run the test suite |

Scraper options: `--no-cache`, `--reset`, `--models`, `--delay`, `-v`.

---

## API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web interface |
| `GET` | `/health` · `/stats` | Service readiness · dataset statistics |
| `POST` | `/chat` | Ask a natural-language question |
| `GET` | `/phones` · `/phones/search?q=` · `/phones/{id}` | Browse, search, full specifications |
| `POST` | `/compare` | Compare two or three phones |
| `GET` | `/agents` | Describe the agent crew |
| `POST` | `/reviews` | Generate a review |

```bash
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "{\"question\": \"Which Samsung phone has the best battery life?\"}"
```

```json
{
  "answer": "Samsung Galaxy S25 Ultra has the best battery life with an active use score of 14:49h.",
  "intent": "superlative",
  "sources": [{ "phone": "Samsung Galaxy S25 Ultra", "section": "Our Tests", "score": 0.93 }],
  "elapsed_seconds": 10.65
}
```

---

## Database

Four tables: **`phones`** · **`specifications`** · **`prices`** · **`spec_categories`**.

The scraped data is stored in two complementary forms. `specifications` keeps every
published row verbatim so nothing is lost, while `phones` keeps the *parsed* headline values
in typed numeric columns. That second form turns "which phone has the biggest battery?" into
a single `ORDER BY` rather than something the language model has to infer.

---

## Design notes

**Retrieval is hybrid and grounded in SQL.** Phone questions contain exact tokens
(`Snapdragon 8 Gen 2`, `120Hz`) that embeddings blur, so dense vectors are fused with BM25.
Questions are then classified — a *superlative* ("best battery life") is answered from a SQL
ranking, and a *comparison* from a verified side-by-side table. The model reports results
that have already been computed; it never produces a ranking itself.

**Battery life is measured, not inferred.** Seven phones share a 5000 mAh cell, so capacity
cannot rank them. GSMArena's measured test result is parsed into a numeric column — but that
test has used two incompatible scales, so both the value and its metric are stored, and
rankings never mix them.

**Reviews are composed section by section.** A 1.5B model asked for a whole review drifts,
so the review agent issues one tightly-scoped brief per section containing only the relevant
facts. Two of the three agents use no language model at all, so every figure in a review
comes from the database.

**Data fidelity.** GSMArena separates camera lenses with `<br>` inside one table cell.
Collapsing that to plain text merges three lenses into one string; the scraper converts the
breaks to an explicit separator so each lens stays distinct throughout the system.

---

## Testing

```bash
pytest                  # 137 tests, ~5 minutes
pytest -m "not llm"     # 113 tests, ~11 seconds
```

Parsing tests run against committed HTML fixtures, so they need no network or database.
Tests requiring MySQL skip automatically when no server is reachable.

---

## Structure

```
scripts/     Entry points for each stage
src/
  config.py  Central settings
  database/  ORM models, sessions, repository
  scraper/   HTTP client, parsers, GSMArena scraper, pipeline
  rag/       Documents, embeddings, vector store, query analyser, chatbot
  agents/    Base contract, three agents, orchestration
  api/       FastAPI app, schemas, web client
tests/       137 tests
data/        Scraped dataset
```

---

## Data source

Specifications are scraped from [GSMArena](https://www.gsmarena.com/). The scraper requests
only pages permitted by `robots.txt`, waits two seconds between requests, and caches every
response. Data is used for educational and demonstration purposes.

# ScholarStream — Project Memory

> **Purpose of this file:** Single source of truth for any developer or agent joining this project. Read this before touching any code. Updated with every meaningful change.

---

## What Is ScholarStream?

ScholarStream is a **production-ready RAG (Retrieval-Augmented Generation) ingestion pipeline for scientific papers**. It automatically fetches papers from arXiv, parses their full text using GPU-accelerated PDF analysis, stores structured data in PostgreSQL, indexes it in OpenSearch for BM25 search, and will ultimately serve answers via a local LLM (Ollama) through a FastAPI interface.

Think of it as: **arXiv → PDF Parser → PostgreSQL + OpenSearch → LLM → FastAPI**

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Language | Python 3.12 | Runtime |
| Package Manager | `uv` | Dependency management (replaces pip/venv) |
| API Framework | FastAPI + Uvicorn | REST API server |
| PDF Parsing | Docling | GPU-accelerated structured PDF → Markdown |
| LLM Inference | Ollama + langchain-ollama | Local LLM (no API costs) |
| RAG Framework | LangChain + langchain-community | Chunking, retrieval, chains |
| Metadata DB | PostgreSQL 16 (psycopg2) | Paper metadata + full text storage |
| Search / Vector DB | OpenSearch 2.11.0 (opensearch-py) | BM25 full-text search, future vector search |
| HTTP Client | httpx | Async PDF downloads |
| Workflow Scheduler | Apache Airflow 3.1.7 | Automated periodic ingestion |
| Containerization | Docker + `compose.yml` | Full service orchestration |
| GPU | NVIDIA CUDA 12.4.1 (RTX 5070) | Docling PDF parsing acceleration |

---

## Repository Structure

```
ScholarStream/
├── main.py               # FastAPI app entry point
├── arxiv_client.py       # arXiv metadata fetcher + PDF downloader + DB saver
├── pdf_parser.py         # Docling-based PDF → Markdown parser
├── ingest.py             # End-to-end pipeline orchestrator (run this)
├── init_db.py            # PostgreSQL table creation
├── opensearch_service.py # OpenSearch client: health check, index creation, indexing
├── init_os.py            # One-shot OpenSearch index setup script
├── compose.yml           # Full Docker service stack
├── Dockerfile            # CUDA-based container image for the API/app
├── pyproject.toml        # Project metadata + dependencies (uv)
├── memory.md             # ← You are here
└── data/
    └── pdfs/             # Local PDF cache (gitignored in practice)
```

---

## Infrastructure (Docker Services)

All services are defined in `compose.yml` and launched with `docker compose up`.

| Service | Container | Port | Notes |
|---|---|---|---|
| PostgreSQL 16 | `ss-postgres` | `5432` | DB: `rag_db`, User: `rag_user`, Pass: `rag_password` |
| OpenSearch 2.11.0 | `ss-opensearch` | `9200` | Security disabled, single-node, 512MB heap |
| OpenSearch Dashboards | `ss-dashboards` | `5601` | UI for OpenSearch |
| Ollama | `ss-ollama` | `11434` | GPU-accelerated local LLM runner |
| Apache Airflow 3.1.7 | `ss-airflow` | `8080` | Built from same `Dockerfile` as API; standalone mode; GPU-enabled; `AIRFLOW_HOME=/opt/airflow`; DAGs mounted at `./dags` |
| FastAPI App | `ss-api` | `8000` | Built from `Dockerfile`, GPU-enabled, hot-reloaded via volume mount |

**Named volumes:** `postgres_data`, `opensearch_data`, `ollama_data`

> **Note:** The Python virtual environment lives at `/opt/venv` inside the container (baked into the image). There is no `venv_data` volume — the venv is not bind-mounted.

---

## Database Schema

**PostgreSQL — `papers` table** (created by `init_db.py`):

```sql
CREATE TABLE IF NOT EXISTS papers (
    id             SERIAL PRIMARY KEY,
    title          TEXT NOT NULL,
    authors        TEXT[],
    summary        TEXT,
    arxiv_id       VARCHAR(20) UNIQUE,   -- arXiv paper ID, natural key
    published_date TIMESTAMP,
    pdf_url        TEXT,
    content        TEXT,                 -- Full parsed Markdown content from Docling
    processed_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

Upsert strategy: `ON CONFLICT (arxiv_id) DO UPDATE SET content, processed_at` — safe to re-run.

---

## OpenSearch Index

**Index name:** `arxiv-papers` (created by `init_os.py` via `opensearch_service.py`)

```
Shards: 1 | Replicas: 0 (single-node dev setup)

Field mappings:
  title          → text  (standard analyzer)
  authors        → keyword
  summary        → text
  content        → text
  arxiv_id       → keyword (document ID)
  published_date → date
  pdf_url        → keyword
```

---

## Module Responsibilities

### `arxiv_client.py` — `ArxivClient`
- `fetch_papers(query, max_results, from_date, to_date)` — async, wraps synchronous arxiv SDK in executor. Returns list of paper dicts.
- `download_pdf(paper_dict)` — async, downloads to `data/pdfs/{arxiv_id}.pdf` with local cache check (idempotent).
- `save_to_db(papers)` — module-level async function, upserts list of paper dicts into PostgreSQL.
- Rate limiting: 3s delay between arXiv requests, 3 retries, page size 10.

### `pdf_parser.py` — `PaperParser`
- `parse_pdf(pdf_path)` — synchronous. Uses Docling `DocumentConverter` to convert a PDF file to structured Markdown.
- Output format is Markdown (preserves headings, tables) — optimized for downstream RAG chunking.
- GPU acceleration handled transparently by Docling using the NVIDIA runtime.

### `ingest.py` — `run_pipeline()`
- Full end-to-end pipeline in one async function:
  1. `init_db()` — ensures table exists
  2. `ArxivClient.fetch_papers()` — fetch metadata
  3. `ArxivClient.download_pdf()` — download PDF
  4. `PaperParser.parse_pdf()` — extract Markdown text
  5. `save_to_db()` — persist to PostgreSQL
- Currently configured to fetch 1 paper for `"Algorithmic Reasoning"` query as integration test.

### `init_db.py` — `init_db()`
- Idempotent (`CREATE TABLE IF NOT EXISTS`). Run once or on every startup.

### `opensearch_service.py` — `OpenSearchService`
- `health_check()` — pings OpenSearch, returns bool.
- `create_index()` — creates `arxiv-papers` index if not exists. Returns `True` if created, `False` if already existed.
- `index_paper(paper_dict)` — indexes a single paper using `arxiv_id` as document ID.

### `init_os.py`
- One-shot script: connects to OpenSearch, runs health check, creates the index.

### `main.py`
- FastAPI app skeleton. Single `GET /` endpoint returning `{"status": "ScholarStream API is running"}`.
- Entry point for the `ss-api` Docker container via `uv run uvicorn main:app`.

---

## Environment Variables

| Variable | Default | Used By |
|---|---|---|
| `DATABASE_URL` | `postgresql://rag_user:rag_password@postgres:5432/rag_db` | `arxiv_client.py`, `init_db.py` |
| `OPENSEARCH__HOST` | `http://opensearch:9200` | `opensearch_service.py` |
| `OPENSEARCH__INDEX_NAME` | `arxiv-papers` | `opensearch_service.py` |
| `OPENSEARCH_URL` | `http://opensearch:9200` | Set in `compose.yml` for the API container |

---

## How to Run

```bash
# 1. Start all infrastructure
docker compose up -d

# 2. (First time only) Initialize OpenSearch index
/opt/venv/bin/python init_os.py

# 3. Run the full ingestion pipeline
/opt/venv/bin/python ingest.py

# 4. Start the API server (also started automatically by Docker)
/opt/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000

# 5. Run Airflow (also started automatically by Docker)
/opt/venv/bin/airflow standalone
```

> **Note:** Services use direct venv binaries (`/opt/venv/bin/...`) — not `uv run`. This applies both inside Docker containers and when running locally against the same venv.

---

## Progress Log

### Phase 1 — Infrastructure Setup
- Designed and configured full Docker Compose stack: PostgreSQL, OpenSearch, OpenSearch Dashboards, Ollama, Airflow, FastAPI.
- Created NVIDIA CUDA 12.4.1-based `Dockerfile` for GPU-accelerated PDF parsing.
- Set up Python 3.12 project with `uv` as package manager (`pyproject.toml`).
- Configured all named volumes and service health checks.

### Phase 2 — Data Ingestion Pipeline
- Built `ArxivClient` with async paper fetching, local PDF caching, and rate-limit compliance.
- Built `PaperParser` using Docling for GPU-accelerated PDF → Markdown conversion.
- Built `init_db.py` for idempotent PostgreSQL schema initialization.
- Built `save_to_db()` with upsert strategy on `arxiv_id`.
- Integrated everything in `ingest.py` as a single runnable pipeline.
- Verified end-to-end: fetch → download → parse → store.

### Phase 3 — OpenSearch Indexing Layer
- Built `OpenSearchService` with health check, index creation (BM25 mappings), and paper indexing.
- Built `init_os.py` as one-shot index setup script.
- FastAPI app skeleton (`main.py`) created as the eventual query-serving layer.

### Phase 4 — Infrastructure Hardening (2026-03-08)
- Upgraded Apache Airflow from `2.7.1` to `3.1.7`.
- Airflow now runs from the same `scholar-stream-app:latest` image (built via `Dockerfile`) instead of the official `apache/airflow` image — unified dependency management.
- Virtual environment path changed from `/app/.venv` to `/opt/venv` (baked into image, not a Docker volume).
- All service commands switched from `uv run <tool>` to direct binary invocations (`/opt/venv/bin/uvicorn`, `/opt/venv/bin/airflow`) for reliability and explicit path control.
- Added healthcheck to OpenSearch service (HTTP probe on `/_cluster/health`).
- Airflow container now runs as root (`user: 0:0`) and is GPU-enabled (RTX 5070).
- Added `AIRFLOW_HOME=/opt/airflow` and `./dags:/opt/airflow/dags` volume mount to Airflow service.
- Removed `venv_data` named volume (no longer needed).
- API `depends_on` opensearch now waits for `service_healthy` instead of `service_started`.

---

## What Comes Next (Planned)

- [ ] **Chunking strategy** — split `content` (Markdown) into semantic chunks for RAG.
- [ ] **Vector embeddings** — embed chunks and store in OpenSearch `knn_vector` field.
- [ ] **RAG query pipeline** — LangChain chain: query → OpenSearch retrieval → Ollama LLM → answer.
- [ ] **FastAPI endpoints** — `POST /search`, `POST /ask` for full RAG Q&A.
- [ ] **Airflow DAG** — scheduled nightly ingestion of new arXiv papers.
- [ ] **OpenSearch → index pipeline integration** — after DB save, also call `index_paper()`.

---

## Key Design Decisions

1. **Markdown as intermediate format** — Docling output is stored as Markdown (not raw text) because it preserves document structure (headings, tables), which improves RAG chunking quality.
2. **PostgreSQL as source of truth** — Full paper content lives in PostgreSQL. OpenSearch is a search index, not the primary store.
3. **Local PDF cache** — `data/pdfs/` prevents redundant downloads on re-runs. Idempotent by design.
4. **Upsert on `arxiv_id`** — Re-running the pipeline on the same paper updates `content` and `processed_at` without duplicating rows.
5. **GPU for Docling** — The Docker image is built on `nvidia/cuda:12.4.1-runtime-ubuntu22.04` so Docling can leverage the RTX 5070 for layout analysis.
6. **`uv` over pip** — Faster installs, lockfile-based reproducibility, no separate venv management needed.
7. **`/opt/venv` as the canonical venv path** — The virtual environment is built into the image at `/opt/venv`, not mounted as a volume. This keeps the image self-contained and avoids venv drift across container restarts.
8. **Direct binary calls over `uv run`** — Container commands use `/opt/venv/bin/<tool>` directly for deterministic execution, removing the `uv` runtime as a dependency at launch time.

---

## Keeping This File Up to Date

**This file must be updated whenever any of the following occur:**

- A new file or module is added → add it to the Repository Structure and Module Responsibilities sections.
- A new dependency is added to `pyproject.toml` → update the Tech Stack table.
- The database schema changes → update the Database Schema section.
- The OpenSearch index mapping changes → update the OpenSearch Index section.
- A new Docker service is added or an existing one is modified → update the Infrastructure table.
- A new environment variable is introduced → update the Environment Variables table.
- A planned item from "What Comes Next" is implemented → move it to the Progress Log with a brief description and remove it from the checklist.
- Any key architectural or design decision is made → add it to the Key Design Decisions section.

**Rule:** If you made a change to the codebase and did not update `memory.md`, the change is incomplete. Treat this file as part of every PR or task.

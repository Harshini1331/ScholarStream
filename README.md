# ScholarStream

A production-ready RAG (Retrieval-Augmented Generation) ingestion pipeline for scientific papers. ScholarStream automates the process of fetching academic papers from arXiv, parsing PDFs with GPU acceleration, and storing structured content in PostgreSQL and OpenSearch for intelligent retrieval.

## Features

- **arXiv Integration** - Fetch paper metadata and PDFs directly from arXiv with rate limiting and caching
- **GPU-Accelerated PDF Parsing** - Extract structured text from PDFs using Docling with NVIDIA GPU support
- **Dual Storage Architecture** - PostgreSQL for metadata persistence, OpenSearch for full-text search
- **BM25 Search Index** - Optimized OpenSearch mappings for academic paper retrieval
- **Local LLM Support** - Ollama integration for GPU-accelerated inference
- **Workflow Automation** - Apache Airflow for scheduling and orchestrating pipelines
- **FastAPI Backend** - REST API for programmatic access to the pipeline

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│     arXiv       │────▶│  PDF Parser     │────▶│   PostgreSQL    │
│   (Metadata)    │     │   (Docling)     │     │   (Metadata)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                │
                                ▼
                        ┌─────────────────┐
                        │   OpenSearch    │
                        │  (Full-Text)    │
                        └─────────────────┘
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.12 |
| PDF Parsing | Docling (GPU-accelerated) |
| Metadata DB | PostgreSQL 16 |
| Search Engine | OpenSearch 2.11 |
| Local LLM | Ollama |
| Workflow | Apache Airflow |
| API Framework | FastAPI + Uvicorn |
| Package Manager | uv |
| Containerization | Docker + NVIDIA CUDA |

## Prerequisites

- Docker & Docker Compose
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit
- Python 3.12 (for local development)

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Harshini1331/ScholarStream.git
cd ScholarStream
```

### 2. Start Infrastructure Services

```bash
docker compose up -d
```

This starts:
- **PostgreSQL** on port `5432`
- **OpenSearch** on port `9200`
- **OpenSearch Dashboards** on port `5601`
- **Ollama** on port `11434`
- **Airflow** on port `8080`
- **FastAPI** on port `8000`

### 3. Initialize the Database

```bash
docker compose exec api uv run python init_db.py
```

### 4. Initialize OpenSearch Index

```bash
docker compose exec api uv run python init_os.py
```

### 5. Run the Ingestion Pipeline

```bash
docker compose exec api uv run python ingest.py
```

## Project Structure

```
ScholarStream/
├── arxiv_client.py      # arXiv API client with PDF downloading
├── pdf_parser.py        # GPU-accelerated PDF to Markdown conversion
├── opensearch_service.py # OpenSearch indexing and search
├── init_db.py           # PostgreSQL schema initialization
├── init_os.py           # OpenSearch index setup
├── ingest.py            # Main ingestion pipeline orchestrator
├── compose.yml          # Docker Compose configuration
├── Dockerfile           # CUDA-enabled container image
├── pyproject.toml       # Project dependencies
└── README.md
```

## Components

### arXiv Client (`arxiv_client.py`)

Handles interaction with the arXiv API:
- Fetches paper metadata with configurable queries and date ranges
- Downloads PDFs with local caching for idempotency
- Persists data to PostgreSQL with upsert strategy

```python
client = ArxivClient()
papers = await client.fetch_papers(query="cat:cs.AI", max_results=5)
pdf_path = await client.download_pdf(papers[0])
```

### PDF Parser (`pdf_parser.py`)

Converts PDFs to structured Markdown using Docling:
- Preserves document structure (headings, tables)
- Optimized for RAG chunking
- GPU-accelerated processing

```python
parser = PaperParser()
markdown_content = parser.parse_pdf(pdf_path)
```

### OpenSearch Service (`opensearch_service.py`)

Manages the search index:
- BM25-optimized index mappings
- Full-text search on titles, summaries, and content
- Keyword search on authors and arXiv IDs

```python
os_service = OpenSearchService()
os_service.create_index()
os_service.index_paper(paper_dict)
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql://rag_user:rag_password@postgres:5432/rag_db` | PostgreSQL connection string |
| `OPENSEARCH__HOST` | `http://opensearch:9200` | OpenSearch host URL |
| `OPENSEARCH__INDEX_NAME` | `arxiv-papers` | OpenSearch index name |

### Database Schema

```sql
CREATE TABLE papers (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    authors TEXT[],
    summary TEXT,
    arxiv_id VARCHAR(20) UNIQUE,
    published_date TIMESTAMP,
    pdf_url TEXT,
    content TEXT,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Local Development

### Install Dependencies

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv sync
```

### Run Individual Components

```bash
# Test arXiv client
uv run python arxiv_client.py

# Test PDF parser
uv run python pdf_parser.py

# Initialize database
uv run python init_db.py

# Initialize OpenSearch
uv run python init_os.py

# Run full pipeline
uv run python ingest.py
```

## Docker Services

| Service | Port | Description |
|---------|------|-------------|
| `ss-postgres` | 5432 | PostgreSQL database |
| `ss-opensearch` | 9200 | OpenSearch search engine |
| `ss-dashboards` | 5601 | OpenSearch Dashboards UI |
| `ss-ollama` | 11434 | Ollama LLM server |
| `ss-airflow` | 8080 | Airflow web interface |
| `ss-api` | 8000 | FastAPI application |

## GPU Support

The project is optimized for NVIDIA GPUs:

- **Dockerfile** uses `nvidia/cuda:12.4.1-runtime-ubuntu22.04` base image
- **Docling** leverages GPU for document layout analysis
- **Ollama** uses GPU for LLM inference

Ensure the NVIDIA Container Toolkit is installed:

```bash
# Ubuntu/Debian
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## Dependencies

Core Python packages:
- `arxiv` - arXiv API client
- `docling` - PDF parsing and document intelligence
- `fastapi` + `uvicorn` - Web framework
- `langchain` + `langchain-ollama` - LLM orchestration
- `opensearch-py` - OpenSearch client
- `psycopg2-binary` - PostgreSQL adapter
- `httpx` - Async HTTP client

## License

This project is open source. See the repository for license details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

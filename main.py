import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from langchain_ollama import OllamaEmbeddings, ChatOllama
from opensearch_service import OpenSearchService


# ---------------------------------------------------------------------------
# App Lifespan - startup health checks
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os_service.health_check():
        print("WARNING: Could not connect to OpenSearch on startup.")
    yield


# ---------------------------------------------------------------------------
# Service Initialization
# ---------------------------------------------------------------------------

os_service = OpenSearchService()

embeddings_model = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://ollama:11434"
)
llm = ChatOllama(
    model="llama3",
    base_url="http://ollama:11434"
)

app = FastAPI(title="ScholarStream RAG API", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str
    size: int = 10
    from_: int = 0
    arxiv_id_filter: str = None


class AskRequest(BaseModel):
    question: str
    top_k: int = 4


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    """Service health + OpenSearch connectivity check."""
    return {
        "status": "online",
        "opensearch": os_service.health_check()
    }


@app.get("/stats")
async def index_stats():
    """Returns OpenSearch index document count and size."""
    return os_service.get_index_stats()


@app.get("/search")
async def search_get(
    q: str = Query(..., description="Search query string"),
    size: int = Query(10, description="Number of results to return"),
    from_: int = Query(0, description="Pagination offset"),
    arxiv_id: str = Query(None, description="Filter by arxiv_id")
):
    """
    Simple BM25 keyword search via GET.
    Example: GET /search?q=large+language+models&size=5
    Supports short queries: /search?q=AI or /search?q=ML
    """
    try:
        results = os_service.bm25_search(
            query=q,
            size=size,
            from_=from_,
            arxiv_id_filter=arxiv_id
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
async def search_post(request: SearchRequest):
    """
    Advanced BM25 search via POST with full options.
    Supports field boosting (title^3, summary^2, content^1),
    fuzzy matching, highlighting, and pagination.
    """
    try:
        results = os_service.bm25_search(
            query=request.query,
            size=request.size,
            from_=request.from_,
            arxiv_id_filter=request.arxiv_id_filter
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
async def ask_paper(request: AskRequest):
    """
    RAG Q&A endpoint.
    Flow: embed question → KNN search OpenSearch → build prompt → Ollama LLM → answer
    """
    try:
        # 1. Vectorize the incoming question
        query_vector = embeddings_model.embed_query(request.question)

        # 2. KNN vector search in OpenSearch
        search_query = {
            "size": request.top_k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_vector,
                        "k": request.top_k
                    }
                }
            },
            "_source": ["content", "title", "section", "pdf_url"]
        }

        response = os_service.client.search(
            index=os_service.index_name,
            body=search_query
        )
        hits = response["hits"]["hits"]

        if not hits:
            raise HTTPException(
                status_code=404,
                detail="No relevant context found for this question."
            )

        # 3. Build augmented prompt from retrieved chunks
        context_segments = []
        sources = []
        for hit in hits:
            src = hit["_source"]
            source_info = f"Source: {src['title']} (Section: {src['section']})"
            context_segments.append(f"{source_info}\nContent: {src['content']}")
            sources.append(source_info)

        context_text = "\n\n---\n\n".join(context_segments)

        prompt = (
            "You are a research assistant. Answer the question based ONLY on the provided context.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {request.question}\n\n"
            "Answer:"
        )

        # 4. Generate grounded answer via Llama 3
        answer = llm.invoke(prompt)

        return {
            "answer": answer.content,
            "sources": list(set(sources))
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from langchain_ollama import OllamaEmbeddings, ChatOllama
from opensearch_service import OpenSearchService


# ---------------------------------------------------------------------------
# App Lifespan
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
# Request Models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str
    size: int = 10
    from_: int = 0
    arxiv_id_filter: str = None


class HybridSearchRequest(BaseModel):
    query: str
    size: int = 10
    rrf_k: int = 60          # RRF constant — 60 is standard from original paper


class AskRequest(BaseModel):
    question: str
    top_k: int = 4
    use_hybrid: bool = True  # Use hybrid search for context retrieval by default


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    """Service health + OpenSearch connectivity."""
    return {
        "status": "online",
        "opensearch": os_service.health_check()
    }


@app.get("/stats")
async def index_stats():
    """OpenSearch index document count and size."""
    return os_service.get_index_stats()


@app.get("/search")
async def search_get(
    q: str = Query(..., description="Search query string"),
    size: int = Query(10, description="Number of results"),
    from_: int = Query(0, description="Pagination offset"),
    arxiv_id: str = Query(None, description="Filter by arxiv_id")
):
    """
    Simple BM25 keyword search via GET.
    Example: GET /search?q=large+language+models&size=5
    Supports short queries: /search?q=AI
    """
    try:
        return os_service.bm25_search(
            query=q,
            size=size,
            from_=from_,
            arxiv_id_filter=arxiv_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
async def search_post(request: SearchRequest):
    """
    Advanced BM25 search via POST.
    Field boosting: title^3, summary^2, content^1.
    Supports fuzzy matching, highlighting, pagination.
    """
    try:
        return os_service.bm25_search(
            query=request.query,
            size=request.size,
            from_=request.from_,
            arxiv_id_filter=request.arxiv_id_filter
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/hybrid-search")
async def hybrid_search(request: HybridSearchRequest):
    """
    Hybrid search using manual RRF (Reciprocal Rank Fusion).
    Combines BM25 keyword precision + KNN semantic recall.

    Both searches run in parallel, then results are fused using:
        rrf_score = 1/(rrf_k + bm25_rank) + 1/(rrf_k + knn_rank)

    Docs appearing in both result sets get the highest scores.
    rrf_k=60 is the standard constant from the original RRF paper.
    """
    try:
        # Embed the query for KNN search
        query_vector = embeddings_model.embed_query(request.query)

        return os_service.hybrid_search(
            query=request.query,
            query_vector=query_vector,
            size=request.size,
            rrf_k=request.rrf_k
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
async def ask_paper(request: AskRequest):
    """
    RAG Q&A endpoint.

    Flow:
      1. Embed question via nomic-embed-text
      2. Retrieve context via hybrid search (RRF) or KNN only
      3. Build augmented prompt with retrieved chunks
      4. Generate grounded answer via Llama 3
    """
    try:
        query_vector = embeddings_model.embed_query(request.question)

        if request.use_hybrid:
            # Use hybrid RRF search for best context retrieval
            hybrid_results = os_service.hybrid_search(
                query=request.question,
                query_vector=query_vector,
                size=request.top_k
            )
            hits = hybrid_results["results"]
            context_chunks = [
                {
                    "_source": {
                        "title": h["title"],
                        "section": h["section"],
                        "content": h.get("summary") or ""
                    }
                }
                for h in hits
            ]
        else:
            # Fallback: pure KNN search
            knn_body = {
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
                body=knn_body
            )
            context_chunks = response["hits"]["hits"]

        if not context_chunks:
            raise HTTPException(
                status_code=404,
                detail="No relevant context found for this question."
            )

        # Build augmented prompt
        context_segments = []
        sources = []
        for hit in context_chunks:
            src = hit["_source"]
            source_info = f"Source: {src.get('title')} (Section: {src.get('section')})"
            context_segments.append(f"{source_info}\nContent: {src.get('content', '')}")
            sources.append(source_info)

        context_text = "\n\n---\n\n".join(context_segments)
        prompt = (
            "You are a research assistant. Answer the question based ONLY on the provided context.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {request.question}\n\n"
            "Answer:"
        )

        answer = llm.invoke(prompt)

        return {
            "answer": answer.content,
            "sources": list(set(sources)),
            "search_mode": "hybrid" if request.use_hybrid else "vector"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
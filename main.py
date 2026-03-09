import os
import json
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_ollama import OllamaEmbeddings, ChatOllama
from opensearch_service import OpenSearchService
from cache_service import CacheService
from langfuse_service import TracingService


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os_service.health_check():
        print("WARNING: Could not connect to OpenSearch on startup.")
    yield


os_service = OpenSearchService()
cache = CacheService()
tracer = TracingService()

embeddings_model = OllamaEmbeddings(model="nomic-embed-text", base_url="http://ollama:11434")
llm = ChatOllama(model="llama3", base_url="http://ollama:11434")

app = FastAPI(title="ScholarStream RAG API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/ui")
async def serve_ui():
    return FileResponse("scholarstream_ui.html")


class SearchRequest(BaseModel):
    query: str
    size: int = 10
    from_: int = 0
    arxiv_id_filter: str = None

class HybridSearchRequest(BaseModel):
    query: str
    size: int = 10
    rrf_k: int = 60

class AskRequest(BaseModel):
    question: str
    top_k: int = 4
    use_hybrid: bool = True

class StreamRequest(BaseModel):
    question: str
    top_k: int = 3
    use_hybrid: bool = True


def _retrieve_context(question: str, top_k: int, use_hybrid: bool):
    t0 = time.time()
    query_vector = embeddings_model.embed_query(question)
    embed_time = round(time.time() - t0, 3)

    t1 = time.time()
    if use_hybrid:
        hits = os_service.hybrid_search(query=question, query_vector=query_vector, size=top_k)["results"]
        chunks = [{"title": h["title"], "section": h["section"], "content": h.get("summary") or ""} for h in hits]
    else:
        resp = os_service.client.search(index=os_service.index_name, body={
            "size": top_k,
            "query": {"knn": {"embedding": {"vector": query_vector, "k": top_k}}},
            "_source": ["content", "title", "section"]
        })
        chunks = [{"title": h["_source"].get("title"), "section": h["_source"].get("section"), "content": h["_source"].get("content", "")} for h in resp["hits"]["hits"]]
    retrieve_time = round(time.time() - t1, 3)

    if not chunks:
        return None, [], embed_time, retrieve_time

    segments, sources = [], []
    for c in chunks:
        src = f"{c['title']} (Section: {c['section']})"
        segments.append(f"Source: {src}\nContent: {c['content']}")
        sources.append(src)
    return "\n\n---\n\n".join(segments), list(set(sources)), embed_time, retrieve_time


def _build_prompt(question: str, context: str) -> str:
    return (
        "You are a research assistant. Answer based ONLY on the context below.\n"
        "Be concise — maximum 300 words.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )


@app.get("/health")
async def health_check():
    return {"status": "online", "opensearch": os_service.health_check(), "cache": cache.stats(), "tracing": tracer.enabled}

@app.post("/cache/flush")
async def flush_cache():
    return {"deleted": cache.flush_all()}

@app.get("/stats")
async def index_stats():
    return os_service.get_index_stats()

@app.get("/search")
async def search_get(q: str = Query(...), size: int = Query(10), from_: int = Query(0), arxiv_id: str = Query(None)):
    try:
        return os_service.bm25_search(query=q, size=size, from_=from_, arxiv_id_filter=arxiv_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search_post(request: SearchRequest):
    try:
        return os_service.bm25_search(query=request.query, size=request.size, from_=request.from_, arxiv_id_filter=request.arxiv_id_filter)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/hybrid-search")
async def hybrid_search(request: HybridSearchRequest):
    try:
        qv = embeddings_model.embed_query(request.query)
        return os_service.hybrid_search(query=request.query, query_vector=qv, size=request.size, rrf_k=request.rrf_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
async def ask_paper(request: AskRequest):
    t0 = time.time()
    cached = cache.get(request.question, request.top_k, request.use_hybrid)
    if cached:
        cached["cache_hit"] = True
        tracer.trace_ask(question=request.question, answer=cached["answer"], sources=cached["sources"],
                         top_k=request.top_k, use_hybrid=request.use_hybrid, cache_hit=True,
                         response_time_s=round(time.time() - t0, 3))
        return cached

    try:
        context, sources, embed_t, retrieve_t = _retrieve_context(request.question, request.top_k, request.use_hybrid)
        if not context:
            raise HTTPException(status_code=404, detail="No relevant context found.")

        t_gen = time.time()
        answer = llm.invoke(_build_prompt(request.question, context))
        gen_t = round(time.time() - t_gen, 3)
        total_t = round(time.time() - t0, 3)

        resp = {"answer": answer.content, "sources": sources,
                "search_mode": "hybrid" if request.use_hybrid else "vector",
                "response_time_s": total_t, "cache_hit": False}

        cache.set(request.question, request.top_k, request.use_hybrid, resp)
        tracer.trace_ask(question=request.question, answer=answer.content, sources=sources,
                         top_k=request.top_k, use_hybrid=request.use_hybrid, cache_hit=False,
                         response_time_s=total_t, embed_time_s=embed_t,
                         retrieve_time_s=retrieve_t, generate_time_s=gen_t)
        return resp
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stream")
async def stream_answer(request: StreamRequest):
    t0 = time.time()
    cached = cache.get(request.question, request.top_k, request.use_hybrid)
    if cached:
        tracer.trace_stream(question=request.question, answer=cached["answer"], sources=cached["sources"],
                            top_k=request.top_k, use_hybrid=request.use_hybrid, cache_hit=True,
                            response_time_s=round(time.time() - t0, 3))
        def replay():
            yield f"data: {json.dumps({'token': cached['answer']})}\n\n"
            yield f"data: {json.dumps({'done': True, 'sources': cached['sources'], 'cache_hit': True})}\n\n"
        return StreamingResponse(replay(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    try:
        context, sources, _, _ = _retrieve_context(request.question, request.top_k, request.use_hybrid)
        if not context:
            raise HTTPException(status_code=404, detail="No relevant context found.")

        prompt = _build_prompt(request.question, context)

        def token_generator():
            tokens = []
            try:
                for chunk in llm.stream(prompt):
                    if chunk.content:
                        tokens.append(chunk.content)
                        yield f"data: {json.dumps({'token': chunk.content})}\n\n"

                answer = "".join(tokens)
                total_t = round(time.time() - t0, 3)
                cache.set(request.question, request.top_k, request.use_hybrid,
                          {"answer": answer, "sources": sources,
                           "search_mode": "hybrid" if request.use_hybrid else "vector",
                           "response_time_s": total_t})
                tracer.trace_stream(question=request.question, answer=answer, sources=sources,
                                    top_k=request.top_k, use_hybrid=request.use_hybrid,
                                    cache_hit=False, response_time_s=total_t)
                yield f"data: {json.dumps({'done': True, 'sources': sources, 'cache_hit': False})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(token_generator(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------------------------
# Agentic RAG
# ---------------------------------------------------------------------------

import agentic_rag as _agentic_rag

# Agentic RAG uses llama3 (no tool calling needed)
_agentic_rag.init_services(embeddings_model, llm, os_service)


class AgenticAskRequest(BaseModel):
    query: str
    top_k: int = 3
    use_hybrid: bool = True


@app.post("/ask-agentic")
async def ask_agentic(request: AgenticAskRequest):
    """
    Agentic RAG endpoint using LangGraph.

    The agent decides whether to:
    - Respond directly (simple/off-topic questions)
    - Retrieve papers → grade relevance → generate answer
    - Rewrite query and retry if documents aren't relevant

    Returns reasoning_steps showing the agent's decision trail.
    """
    try:
        t0 = time.time()

        # Check cache first
        cached = cache.get(request.query, request.top_k, request.use_hybrid)
        if cached and "reasoning_steps" in cached:
            cached["cache_hit"] = True
            return cached

        result = _agentic_rag.run_agentic_rag(query=request.query, top_k=request.top_k)
        result["query"] = request.query
        result["response_time_s"] = round(time.time() - t0, 3)
        result["cache_hit"] = False

        # Cache the result
        cache.set(request.query, request.top_k, request.use_hybrid, result)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
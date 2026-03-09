from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_ollama import OllamaEmbeddings, ChatOllama
from opensearch_service import OpenSearchService
from contextlib import asynccontextmanager
import os

# 1. Lifespan Handler: Manages startup and shutdown cleanly
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup Logic
    print("🚀 ScholarStream API Starting Up...")
    try:
        if os_service.client.ping():
            print("✅ OpenSearch Connection: SUCCESS")
        else:
            print("❌ OpenSearch Connection: FAILED")
    except Exception as e:
        print(f"⚠️ OpenSearch Connectivity Error: {e}")
    
    yield  # The API runs while this is suspended
    
    # Shutdown Logic
    print("🛑 ScholarStream API Shutting Down...")

app = FastAPI(title="ScholarStream RAG API", lifespan=lifespan)

# Setup Models and Services
embeddings_model = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://ollama:11434"
)

llm = ChatOllama(
    model="llama3",
    base_url="http://ollama:11434"
)

os_service = OpenSearchService()

class QueryRequest(BaseModel):
    question: str
    top_k: int = 4

@app.get("/")
async def root():
    """Health check endpoint to verify routing is working."""
    return {
        "status": "online",
        "service": "ScholarStream RAG",
        "endpoints": ["/ask", "/docs"]
    }

@app.post("/ask")
async def ask_paper(request: QueryRequest):
    # 1. Vectorize the user's question
    query_vector = embeddings_model.embed_query(request.question)

    # 2. Search OpenSearch (Using 'content' field to match your index)
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
        "_source": ["content", "title", "section"] # Changed 'text' to 'content'
    }
    
    try:
        response = os_service.client.search(
            index=os_service.index_name,
            body=search_query
        )
        
        hits = response['hits']['hits']
        if not hits:
            raise HTTPException(status_code=404, detail="No relevant context found in papers.")

        # 3. Construct the Context from the 'content' field
        context_parts = []
        sources = []
        for hit in hits:
            source_info = hit['_source']
            context_parts.append(source_info.get('content', ''))
            sources.append(f"{source_info.get('title')} (Section: {source_info.get('section')})")

        context = "\n---\n".join(context_parts)

        # 4. Grounded Generation
        prompt = f"""
        You are an expert research assistant. Answer the question based ONLY on the provided context from scientific papers. 
        If the answer isn't in the context, say you don't know.
        
        Context:
        {context}
        
        Question: {request.question}
        
        Answer:"""

        answer = llm.invoke(prompt)

        return {
            "answer": answer.content,
            "sources": list(set(sources)) 
        }
    except Exception as e:
        print(f"Error during RAG query: {e}")
        raise HTTPException(status_code=500, detail=str(e))
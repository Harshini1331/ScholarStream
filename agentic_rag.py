"""
Agentic RAG using LangGraph — no tool calling required.

Workflow:
  START
    ↓
  decide
    ├─ direct → END
    └─ retrieve
         ↓
       grade_documents
         ├─ relevant → generate_answer → END
         └─ not relevant → rewrite_query → retrieve (max 2 retries)
"""

import os
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END, START

from agentic_prompts import DECIDE_PROMPT, GRADE_PROMPT, REWRITE_PROMPT, GENERATE_PROMPT

MAX_RETRIES = 2

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    query: str
    context: str
    sources: list
    reasoning_steps: list
    retrieval_attempts: int
    answer: str
    route: str  # "direct" or "retrieve"

# ---------------------------------------------------------------------------
# Service references
# ---------------------------------------------------------------------------

_embeddings = None
_llm = None
_os_service = None


def init_services(embeddings, llm, os_service):
    global _embeddings, _llm, _os_service
    _embeddings = embeddings
    _llm = llm
    _os_service = os_service


# ---------------------------------------------------------------------------
# Retrieval helper
# ---------------------------------------------------------------------------

def _retrieve(query: str, top_k: int = 3) -> tuple[str, list]:
    """Fetch and format chunks from OpenSearch. Returns (context, sources)."""
    try:
        vector = _embeddings.embed_query(query)
        results = _os_service.hybrid_search(query=query, query_vector=vector, size=top_k)
        hits = results.get("results", [])
        if not hits:
            return "", []
        parts, sources = [], []
        for h in hits:
            title = h.get("title", "Unknown")
            section = h.get("section", "")
            content = h.get("summary") or ""
            parts.append(f"[Title: {title} | Section: {section}]\n{content}")
            sources.append(f"{title} (Section: {section})")
        return "\n\n---\n\n".join(parts), list(set(sources))
    except Exception as e:
        return "", []


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def decide(state: AgentState) -> AgentState:
    """LLM decides: answer directly or retrieve papers."""
    prompt = DECIDE_PROMPT.format(query=state["query"])
    response = _llm.invoke(prompt)
    text = response.content.strip().lower()

    # If the response looks like an answer rather than a decision, go direct
    needs_retrieval = any(kw in text for kw in [
        "search", "retrieve", "look up", "find papers", "check database",
        "need to search", "will search", "let me search"
    ])

    route = "retrieve" if needs_retrieval else "direct"
    steps = state.get("reasoning_steps", [])

    if route == "direct":
        return {
            "route": "direct",
            "answer": response.content,
            "reasoning_steps": steps + ["Responded directly without retrieval"],
        }
    else:
        return {
            "route": "retrieve",
            "reasoning_steps": steps + ["Decided to retrieve relevant papers"],
        }


def retrieve(state: AgentState) -> AgentState:
    """Fetch documents from OpenSearch."""
    context, sources = _retrieve(state["query"])
    steps = state.get("reasoning_steps", [])
    if context:
        steps = steps + [f"Retrieved documents from database"]
    else:
        steps = steps + ["No documents found in database"]
    return {
        "context": context,
        "sources": sources,
        "reasoning_steps": steps,
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
    }


def grade_documents(state: AgentState) -> AgentState:
    """Grade retrieved chunks for relevance."""
    context = state.get("context", "")
    if not context:
        return {
            "context": "",
            "reasoning_steps": state.get("reasoning_steps", []) + ["No documents to grade"],
        }

    chunks = context.split("\n\n---\n\n")
    relevant = []
    for chunk in chunks:
        prompt = GRADE_PROMPT.format(query=state["query"], chunk=chunk[:500])
        verdict = _llm.invoke(prompt).content.strip().lower()
        if "yes" in verdict:
            relevant.append(chunk)

    steps = state.get("reasoning_steps", [])
    if relevant:
        steps = steps + [f"Graded {len(relevant)}/{len(chunks)} chunks as relevant"]
    else:
        steps = steps + ["Retrieved documents were not relevant"]

    return {
        "context": "\n\n---\n\n".join(relevant),
        "reasoning_steps": steps,
    }


def rewrite_query(state: AgentState) -> AgentState:
    """Rewrite the query for better retrieval."""
    prompt = REWRITE_PROMPT.format(query=state["query"])
    new_query = _llm.invoke(prompt).content.strip()
    steps = state.get("reasoning_steps", []) + [f"Rewrote query to: '{new_query}'"]
    return {
        "query": new_query,
        "reasoning_steps": steps,
    }


def generate_answer(state: AgentState) -> AgentState:
    """Generate final answer from context."""
    context = state.get("context", "")
    if not context:
        prompt = f"Answer this question concisely: {state['query']}"
    else:
        prompt = GENERATE_PROMPT.format(query=state["query"], context=context)

    answer = _llm.invoke(prompt).content
    steps = state.get("reasoning_steps", []) + ["Generated answer from relevant documents"]
    return {"answer": answer, "reasoning_steps": steps}


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def route_after_decide(state: AgentState) -> str:
    return "end" if state.get("route") == "direct" else "retrieve"


def route_after_grading(state: AgentState) -> str:
    if state.get("context"):
        return "generate_answer"
    if state.get("retrieval_attempts", 0) >= MAX_RETRIES:
        return "generate_answer"
    return "rewrite_query"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("decide", decide)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("generate_answer", generate_answer)

    graph.add_edge(START, "decide")
    graph.add_conditional_edges("decide", route_after_decide,
                                {"end": END, "retrieve": "retrieve"})
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges("grade_documents", route_after_grading,
                                {"generate_answer": "generate_answer",
                                 "rewrite_query": "rewrite_query"})
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate_answer", END)

    return graph.compile()


_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_agentic_rag(query: str, top_k: int = 3) -> dict:
    graph = get_graph()
    final = graph.invoke({
        "query": query,
        "context": "",
        "sources": [],
        "reasoning_steps": [],
        "retrieval_attempts": 0,
        "answer": "",
        "route": "",
    })
    return {
        "answer": final.get("answer", ""),
        "sources": final.get("sources", []),
        "reasoning_steps": final.get("reasoning_steps", []),
        "retrieval_attempts": final.get("retrieval_attempts", 0),
        "search_mode": "hybrid",
    }
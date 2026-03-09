"""
Prompt templates for the Agentic RAG LangGraph workflow.
"""

DECIDE_PROMPT = """You are a research assistant with access to a database of scientific papers.

Given a user query, decide whether you need to search the paper database or can answer directly.

- If the query is about research topics, machine learning, AI, computer vision, NLP, or any technical/scientific subject: USE the search tool.
- If the query is simple math, general knowledge, greetings, or clearly unrelated to research: RESPOND DIRECTLY without searching.

Query: {query}"""

GRADE_PROMPT = """You are grading whether a retrieved document chunk is relevant to a user query.

Query: {query}

Retrieved chunk:
{chunk}

Is this chunk relevant to answering the query? Reply with only "yes" or "no"."""

REWRITE_PROMPT = """The retrieved documents were not relevant enough to answer this query.
Rewrite the query to be more specific and likely to retrieve relevant scientific papers.

Original query: {query}

Rewritten query (just the query text, nothing else):"""

GENERATE_PROMPT = """You are a research assistant. Answer the question based ONLY on the provided context.
Be concise — maximum 300 words.

Context:
{context}

Question: {query}

Answer:"""
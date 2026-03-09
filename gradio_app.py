"""
ScholarStream Gradio Interface
Run with: python gradio_app.py
Opens at: http://localhost:7861
"""

import httpx
import gradio as gr
import json

API_BASE = "http://localhost:8000"


def search_papers(query: str, size: int, search_mode: str):
    """BM25 or Hybrid search and format results."""
    if not query.strip():
        return "Please enter a search query."

    try:
        if search_mode == "Hybrid (BM25 + Vector)":
            response = httpx.post(
                f"{API_BASE}/hybrid-search",
                json={"query": query, "size": int(size)},
                timeout=30
            )
        else:
            response = httpx.post(
                f"{API_BASE}/search",
                json={"query": query, "size": int(size)},
                timeout=30
            )

        data = response.json()
        results = data.get("results", [])
        total = data.get("total", 0)
        mode = data.get("search_mode", "bm25")

        if not results:
            return "No results found."

        output = f"**{total} total matches** (showing {len(results)}) — mode: `{mode}`\n\n"
        output += "---\n\n"

        for i, r in enumerate(results, 1):
            title = r.get("title") or "Unknown Title"
            arxiv_id = r.get("arxiv_id") or ""
            section = r.get("section") or ""
            pdf_url = r.get("pdf_url") or ""

            if mode == "hybrid":
                score_info = f"RRF: `{r.get('rrf_score', 0):.4f}` | BM25 rank: {r.get('bm25_rank')} | KNN rank: {r.get('knn_rank')}"
            else:
                score_info = f"BM25 score: `{r.get('bm25_score', 0):.4f}`"

            output += f"### {i}. {title}\n"
            output += f"- **arXiv ID:** `{arxiv_id}` | **Section:** {section}\n"
            output += f"- **Score:** {score_info}\n"
            if pdf_url:
                output += f"- **PDF:** [{pdf_url}]({pdf_url})\n"

            # Show highlights if available
            highlights = r.get("highlights", {})
            if highlights:
                output += "- **Highlights:**\n"
                for field, snippets in highlights.items():
                    for snippet in snippets[:1]:
                        output += f"  - *{field}*: {snippet}\n"

            output += "\n"

        return output

    except Exception as e:
        return f"Error: {str(e)}"


def ask_question(question: str, top_k: int, use_hybrid: bool):
    """Non-streaming RAG Q&A."""
    if not question.strip():
        return "Please enter a question.", ""

    try:
        response = httpx.post(
            f"{API_BASE}/ask",
            json={
                "question": question,
                "top_k": int(top_k),
                "use_hybrid": use_hybrid
            },
            timeout=120
        )
        data = response.json()
        answer = data.get("answer", "No answer generated.")
        sources = data.get("sources", [])
        mode = data.get("search_mode", "unknown")

        sources_text = f"**Search mode:** `{mode}`\n\n**Sources:**\n"
        for s in sources:
            sources_text += f"- {s}\n"

        return answer, sources_text

    except Exception as e:
        return f"Error: {str(e)}", ""


def stream_question(question: str, top_k: int, use_hybrid: bool):
    """Streaming RAG Q&A — yields tokens as they arrive."""
    if not question.strip():
        yield "Please enter a question."
        return

    accumulated = ""
    sources_text = ""

    try:
        with httpx.stream(
            "POST",
            f"{API_BASE}/stream",
            json={
                "question": question,
                "top_k": int(top_k),
                "use_hybrid": use_hybrid
            },
            timeout=120
        ) as response:
            for line in response.iter_lines():
                if line.startswith("data: "):
                    payload = line[6:]
                    try:
                        data = json.loads(payload)
                        if "token" in data:
                            accumulated += data["token"]
                            yield accumulated
                        elif "done" in data:
                            sources = data.get("sources", [])
                            sources_text = "\n\n---\n**Sources:**\n" + "\n".join(f"- {s}" for s in sources)
                            yield accumulated + sources_text
                        elif "error" in data:
                            yield accumulated + f"\n\n**Error:** {data['error']}"
                    except json.JSONDecodeError:
                        continue

    except Exception as e:
        yield f"Error: {str(e)}"


def check_health():
    """Check API and OpenSearch health."""
    try:
        health = httpx.get(f"{API_BASE}/health", timeout=5).json()
        stats = httpx.get(f"{API_BASE}/stats", timeout=5).json()
        return (
            f"API: {health.get('status')}\n"
            f"OpenSearch: {'connected' if health.get('opensearch') else 'disconnected'}\n"
            f"Indexed chunks: {stats.get('document_count', 'N/A')}\n"
            f"Index size: {stats.get('index_size_bytes', 0) / 1024 / 1024:.1f} MB"
        )
    except Exception as e:
        return f"API unreachable: {str(e)}"


# ---------------------------------------------------------------------------
# Gradio UI Layout
# ---------------------------------------------------------------------------

with gr.Blocks(title="ScholarStream", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # ScholarStream
    ### Production RAG System for Scientific Papers
    *arXiv → Docling GPU Parsing → PostgreSQL + OpenSearch → Llama 3*
    """)

    # --- Health Status ---
    with gr.Row():
        health_btn = gr.Button("Check System Status", variant="secondary")
        health_output = gr.Textbox(label="System Status", lines=4, interactive=False)
    health_btn.click(fn=check_health, outputs=health_output)

    gr.Markdown("---")

    # --- Search Tab ---
    with gr.Tab("Search Papers"):
        gr.Markdown("Search indexed papers using BM25 keyword search or Hybrid (BM25 + Vector) search.")

        with gr.Row():
            search_query = gr.Textbox(
                label="Search Query",
                placeholder="e.g. multimodal large language models",
                scale=3
            )
            search_size = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Results", scale=1)

        search_mode = gr.Radio(
            choices=["BM25 Keyword", "Hybrid (BM25 + Vector)"],
            value="Hybrid (BM25 + Vector)",
            label="Search Mode"
        )
        search_btn = gr.Button("Search", variant="primary")
        search_output = gr.Markdown(label="Results")

        search_btn.click(
            fn=search_papers,
            inputs=[search_query, search_size, search_mode],
            outputs=search_output
        )

        gr.Examples(
            examples=[
                ["large language models", 5, "Hybrid (BM25 + Vector)"],
                ["AI", 5, "BM25 Keyword"],
                ["multimodal vision transformer", 5, "Hybrid (BM25 + Vector)"],
                ["surgical reasoning benchmark", 5, "Hybrid (BM25 + Vector)"],
            ],
            inputs=[search_query, search_size, search_mode]
        )

    # --- Streaming Q&A Tab ---
    with gr.Tab("Ask (Streaming)"):
        gr.Markdown("Ask questions about indexed papers. Answers stream in real-time as Llama 3 generates them.")

        stream_question_input = gr.Textbox(
            label="Question",
            placeholder="e.g. What are the key contributions of multimodal LLMs?",
            lines=2
        )

        with gr.Row():
            stream_top_k = gr.Slider(minimum=1, maximum=8, value=3, step=1, label="Chunks to retrieve")
            stream_hybrid = gr.Checkbox(value=True, label="Use Hybrid Search")

        stream_btn = gr.Button("Ask (Streaming)", variant="primary")
        stream_output = gr.Markdown(label="Answer")

        stream_btn.click(
            fn=stream_question,
            inputs=[stream_question_input, stream_top_k, stream_hybrid],
            outputs=stream_output
        )

        gr.Examples(
            examples=[
                ["What are the key contributions of multimodal large language models?", 3, True],
                ["How does BEV representation help in autonomous driving?", 3, True],
                ["What benchmarks are used for surgical AI evaluation?", 3, True],
            ],
            inputs=[stream_question_input, stream_top_k, stream_hybrid]
        )

    # --- Standard Q&A Tab ---
    with gr.Tab("Ask (Standard)"):
        gr.Markdown("Full response with sources. Waits for complete answer before displaying.")

        ask_question_input = gr.Textbox(
            label="Question",
            placeholder="e.g. Explain the diffusion model approach used in Omni-Diffusion",
            lines=2
        )

        with gr.Row():
            ask_top_k = gr.Slider(minimum=1, maximum=8, value=3, step=1, label="Chunks to retrieve")
            ask_hybrid = gr.Checkbox(value=True, label="Use Hybrid Search")

        ask_btn = gr.Button("Ask", variant="primary")

        with gr.Row():
            ask_answer = gr.Textbox(label="Answer", lines=10, interactive=False)
            ask_sources = gr.Markdown(label="Sources")

        ask_btn.click(
            fn=ask_question,
            inputs=[ask_question_input, ask_top_k, ask_hybrid],
            outputs=[ask_answer, ask_sources]
        )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False
    )
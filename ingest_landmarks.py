"""
Ingest landmark papers by specific arXiv ID.
Run with: docker exec ss-api /opt/venv/bin/python ingest_landmarks.py
"""

import asyncio
from ingest import run_pipeline

# Landmark papers by arXiv ID — these are the papers interviewers will ask about
LANDMARK_PAPERS = [
    # Transformers & Attention
    "1706.03762",   # Attention Is All You Need (Vaswani et al.)
    "1810.04805",   # BERT
    "2005.14165",   # GPT-3
    "2302.13971",   # LLaMA
    "2307.09288",   # LLaMA 2
    "2310.06825",   # Mistral 7B

    # Diffusion Models
    "2006.11239",   # DDPM — Denoising Diffusion Probabilistic Models
    "2010.02502",   # Score-based generative models (Song et al.)
    "2112.10752",   # Latent Diffusion Models / Stable Diffusion
    "2204.06125",   # DALL-E 2

    # RAG & Retrieval
    "2005.11401",   # RAG (Lewis et al. — original RAG paper)
    "2208.09257",   # Atlas — few-shot RAG
    "2212.10560",   # Self-RAG

    # Vision Transformers
    "2010.11929",   # ViT — An Image is Worth 16x16 Words
    "2103.14030",   # CLIP (Radford et al.)
    "2301.12597",   # SAM — Segment Anything Model

    # Agents & LangGraph relevant
    "2210.03629",   # ReAct — Reasoning + Acting in LLMs
    "2303.11366",   # HuggingGPT / multi-agent
    "2305.10601",   # Tree of Thoughts
]


async def ingest_by_id(arxiv_id: str):
    """Ingest a single paper by its arXiv ID."""
    try:
        # Use the arxiv ID as query — the pipeline will find and download it
        await run_pipeline(query=arxiv_id, max_results=1)
        print(f"Done: {arxiv_id}")
    except Exception as e:
        print(f"Failed {arxiv_id}: {e}")


async def main():
    print(f"Ingesting {len(LANDMARK_PAPERS)} landmark papers...")
    for paper_id in LANDMARK_PAPERS:
        print(f"\nIngesting: {paper_id}")
        await ingest_by_id(paper_id)
    print("\nAll landmark papers ingested!")


if __name__ == "__main__":
    asyncio.run(main())
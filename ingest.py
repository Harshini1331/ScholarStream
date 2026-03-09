import asyncio
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from arxiv_client import ArxivClient, save_to_db
from pdf_parser import PaperParser
from opensearch_service import OpenSearchService

async def run_pipeline(query="Algorithmic Reasoning"):
    # 1. Setup Infrastructure Clients
    client = ArxivClient()
    parser = PaperParser()
    os_service = OpenSearchService()
    os_service.create_index()
    
    # Initialize Local Embeddings via Ollama on your RTX 5070
    embeddings_model = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://ollama:11434"
    )

    # 2. Ingestion & Parsing
    print(f"Starting ingestion for: {query}")
    papers = await client.fetch_papers(query=query, max_results=1)
    
    for paper in papers:
        # Download PDF and extract structured Markdown using Docling (GPU)
        pdf_path = await client.download_pdf(paper)
        markdown_text = parser.parse_pdf(pdf_path)
        paper['content'] = markdown_text
        
        # Save raw metadata and full text to PostgreSQL
        await save_to_db([paper])

        # 3. Two-Stage Semantic Chunking
        # Stage A: Split by Markdown Headers (Structural Context)
        headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
        header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        header_chunks = header_splitter.split_text(markdown_text)

        # Stage B: Sub-split large headers to fit embedding context (Size Control)
        # Nomic-embed-text likes chunks around 2048 chars; we use 1500 for safety
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=150,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        # Split the structural chunks into size-compliant documents
        final_documents = text_splitter.split_documents(header_chunks)

        print(f"Paper {paper['arxiv_id']} split into {len(final_documents)} sub-chunks.")

        # 4. Vector Embedding & OpenSearch Indexing
        for i, doc in enumerate(final_documents):
            try:
                # Generate 768-dim vector on GPU
                vector = embeddings_model.embed_query(doc.page_content)
                chunk_id = f"{paper['arxiv_id']}_chunk_{i}"
                
                # Extract header metadata preserved by the splitter
                section_name = doc.metadata.get("Header 1") or doc.metadata.get("Header 2", "General")
                
                os_service.index_chunk(
                    chunk_id=chunk_id,
                    paper_metadata={
                        "title": paper['title'],
                        "arxiv_id": paper['arxiv_id'],
                        "section": section_name,
                        "pdf_url": paper['pdf_url']
                    },
                    text=doc.page_content,
                    embedding=vector
                )
            except Exception as e:
                print(f"Error indexing chunk {i} for {paper['arxiv_id']}: {e}")
                continue

    print("Ingestion, Recursive Chunking, and Vector Indexing Complete!")

if __name__ == "__main__":
    asyncio.run(run_pipeline())
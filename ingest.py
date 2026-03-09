import asyncio
import os
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from arxiv_client import ArxivClient, save_to_db
from pdf_parser import PaperParser
from opensearch_service import OpenSearchService

async def run_pipeline(query="Large Language Models", max_results=5):
    # 1. Setup Infrastructure Clients
    client = ArxivClient()
    parser = PaperParser()
    os_service = OpenSearchService()
    
    # Ensure index exists with KNN mappings before we start
    os_service.create_index()
    
    # Initialize Local Embeddings via Ollama
    # base_url points to the 'ollama' container service name
    embeddings_model = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://ollama:11434"
    )

    # 2. Ingestion & Parsing
    print(f"Starting ingestion for: {query}")
    # Fetching 2 papers to demonstrate the batching capability
    papers = await client.fetch_papers(query=query, max_results=max_results)
    
    for paper in papers:
        print(f"Processing: {paper['title']}")
        
        # Download PDF and extract structured Markdown using Docling (GPU Accelerated)
        pdf_path = await client.download_pdf(paper)
        markdown_text = parser.parse_pdf(pdf_path)
        
        # Update paper dict with full content for PostgreSQL storage
        paper['content'] = markdown_text
        
        # Save raw metadata and full text to PostgreSQL (Primary Source of Truth)
        await save_to_db([paper])

        # 3. Two-Stage Semantic Chunking
        # Stage A: Split by Markdown Headers to maintain structural context
        headers_to_split_on = [
            ("#", "Header 1"), 
            ("##", "Header 2"), 
            ("###", "Header 3")
        ]
        header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        header_chunks = header_splitter.split_text(markdown_text)

        # Stage B: Sub-split large sections for the embedding model's context window
        # nomic-embed-text max context is 2048; 1500 chars + 150 overlap is optimal
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=150,
            add_start_index=True,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        # This preserves the 'Header' metadata from the Markdown splitter
        final_documents = text_splitter.split_documents(header_chunks)
        print(f"Paper {paper['arxiv_id']} split into {len(final_documents)} semantic chunks.")

        # 4. GPU Batch Embedding & OpenSearch Indexing
        print(f"Generating embeddings via RTX 5070...")
        
        # Extract text strings for batch processing
        texts_to_embed = [doc.page_content for doc in final_documents]
        
        try:
            # Batch process all chunks in one API call to Ollama
            # This is significantly faster on a GPU than individual calls
            vectors = embeddings_model.embed_documents(texts_to_embed)

            # Zip documents and vectors for concurrent indexing
            for i, (doc, vector) in enumerate(zip(final_documents, vectors)):
                chunk_id = f"{paper['arxiv_id']}_chunk_{i}"
                
                # Extract the highest level header available for section naming
                section_name = (
                    doc.metadata.get("Header 1") or 
                    doc.metadata.get("Header 2") or 
                    doc.metadata.get("Header 3", "General Content")
                )
                
                # Push to OpenSearch KNN index
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
            
            print(f"Successfully indexed {len(final_documents)} vectors for {paper['arxiv_id']}")

        except Exception as e:
            print(f"Error during vectorization/indexing for {paper['arxiv_id']}: {e}")
            continue

    print("\nIngestion, Recursive Chunking, and GPU Batch Indexing Complete!")

if __name__ == "__main__":
    asyncio.run(run_pipeline())
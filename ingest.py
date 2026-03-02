import asyncio
import os
from arxiv_client import ArxivClient, save_to_db
from pdf_parser import PaperParser
from init_db import init_db

async def run_pipeline():
    print("ScholarStream: Integrated Ingestion Pipeline")
    print("=" * 60)

    # 1. Initialize Database Tables
    # This ensures the 'papers' table exists before we try to save
    init_db()

    # 2. Initialize Services
    client = ArxivClient()
    parser = PaperParser()
    
    # 3. Fetch Metadata from ArXiv
    print("Step 1: Querying ArXiv for 'Algorithmic Reasoning'...")
    # Using the correct method name found in your file
    papers = await client.fetch_papers(query="Algorithmic Reasoning", max_results=1)
    
    if not papers:
        print("Error: No papers found.")
        return

    target_paper = papers[0]
    print(f"Found Paper: {target_paper['title']}")

    # 4. Download PDF
    print(f"Step 2: Downloading PDF for {target_paper['arxiv_id']}...")
    pdf_path = await client.download_pdf(target_paper)
    
    if not pdf_path or not pdf_path.exists():
        print("Error: PDF download failed.")
        return

    # 5. Parse PDF using GPU (Docling)
    print(f"Step 3: Extracting structured text via RTX 5070...")
    # parse_pdf is synchronous in your file, so we call it normally
    markdown_content = parser.parse_pdf(pdf_path)
    
    if not markdown_content:
        print("Error: Content extraction failed.")
        return

    # 6. Save to Database
    print("Step 4: Persisting to PostgreSQL...")
    # We add the parsed content to our dictionary before saving
    target_paper['content'] = markdown_content
    await save_to_db([target_paper])

    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION COMPLETE")
    print(f"Paper: {target_paper['title']}")
    print(f"Status: Metadata and Full Text stored in Database")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(run_pipeline())
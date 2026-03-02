import arxiv
import asyncio
import httpx
import psycopg2
import os
from pathlib import Path
from datetime import datetime, timedelta

# Database configuration from Docker environment
DB_URL = os.getenv("DATABASE_URL", "postgresql://rag_user:rag_password@postgres:5432/rag_db")

class ArxivClient:
    def __init__(self, download_dir="data/pdfs"):
        """Initialize the client with rate limiting and local storage."""
        # Official arXiv Client with mandatory 3s delay and 3 retries
        self.client = arxiv.Client(
            page_size=10,
            delay_seconds=3, 
            num_retries=3
        )
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)

    async def fetch_papers(self, query="cat:cs.AI", max_results=5, from_date=None, to_date=None):
        """Fetch paper metadata with optional date range filtering."""
        # Construct date filter: YYYYMMDDHHMM TO YYYYMMDDHHMM
        if from_date and to_date:
            query = f"{query} AND submittedDate:[{from_date} TO {to_date}]"
        
        print(f"Searching arXiv: {query}...")
        
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        # Wrap the synchronous arxiv library in an async executor for performance
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, lambda: list(self.client.results(search)))
        
        papers = []
        for result in results:
            papers.append({
                "title": result.title,
                "authors": [a.name for a in result.authors],
                "summary": result.summary,
                "arxiv_id": result.entry_id.split('/')[-1],
                "published_date": result.published,
                "pdf_url": result.pdf_url
            })
        return papers

    async def download_pdf(self, paper_dict):
        """Downloads PDF with local caching to prevent redundant network calls."""
        arxiv_id = paper_dict['arxiv_id']
        pdf_url = paper_dict['pdf_url']
        file_path = self.download_dir / f"{arxiv_id}.pdf"
        
        # 1. Check Cache: Idempotency check
        if file_path.exists():
            print(f"Cache hit: {arxiv_id}.pdf exists locally.")
            return file_path
        
        # 2. Cache Miss: Async download using httpx
        print(f"Downloading: {pdf_url}...")
        async with httpx.AsyncClient() as client:
            try:
                # follow_redirects is necessary for arXiv's PDF links
                response = await client.get(pdf_url, follow_redirects=True, timeout=30.0)
                if response.status_code == 200:
                    with open(file_path, "wb") as f:
                        f.write(response.content)
                    
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    print(f"Saved to {file_path} ({size_mb:.2f} MB)")
                    return file_path
                else:
                    print(f"Download failed: HTTP {response.status_code}")
                    return None
            except Exception as e:
                print(f"Error downloading {arxiv_id}: {e}")
                return None

async def save_to_db(papers):
    """Persists metadata and full text to PostgreSQL with an upsert strategy."""
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        for paper in papers:
            # We added 'content' to the columns and the values
            # Using 'get' for content to avoid errors if it's missing from the dict
            content_text = paper.get('content', '')

            cur.execute("""
                INSERT INTO papers (title, authors, summary, arxiv_id, published_date, pdf_url, content)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (arxiv_id) 
                DO UPDATE SET 
                    content = EXCLUDED.content,
                    processed_at = CURRENT_TIMESTAMP;
            """, (
                paper['title'], 
                paper['authors'], 
                paper['summary'], 
                paper['arxiv_id'], 
                paper['published_date'], 
                paper['pdf_url'],
                content_text
            ))
            
        conn.commit()
        cur.close()
        conn.close()
        print(f"Successfully synced {len(papers)} papers to database (including full text)!")
    except Exception as e:
        print(f"Database error: {e}")

async def main():
    client = ArxivClient()
    
    # Use a specific ID from 2023 or 2024 to test the download logic
    # '2312.05611' is a well-known paper on RAG
    search = arxiv.Search(id_list=["2312.05611"])
    results = list(client.client.results(search))
    
    if results:
        paper = {
            "title": results[0].title,
            "authors": [a.name for a in results[0].authors],
            "summary": results[0].summary,
            "arxiv_id": results[0].entry_id.split('/')[-1],
            "published_date": results[0].published,
            "pdf_url": results[0].pdf_url
        }
        await save_to_db([paper])
        await client.download_pdf(paper)
'''
async def main():
    """Week 2 Integration Test: Fetch, Download, and Save."""
    client = ArxivClient()
    
    # 1. Test targeted date range (last 48 hours)
    start_date = (datetime.now() - timedelta(days=2)).strftime("%Y%m%d%H%M")
    end_date = datetime.now().strftime("%Y%m%d%H%M")
    
    #results = await client.fetch_papers(from_date=start_date, to_date=end_date, max_results=3)
    results = await client.fetch_papers(max_results=3)   

    if results:
        # 2. Save metadata to DB
        await save_to_db(results)
        
        # 3. Test PDF Download and Caching for the first result
        await client.download_pdf(results[0])
'''

if __name__ == "__main__":
    asyncio.run(main())
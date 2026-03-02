import psycopg2
import os

# Uses the credentials you defined in your final compose.yml
DB_URL = os.getenv("DATABASE_URL", "postgresql://rag_user:rag_password@postgres:5432/rag_db")

def init_db():
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        
        # This matches the 'PaperRepository' requirements for Week 2
        cur.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                id SERIAL PRIMARY KEY,
                title TEXT NOT NULL,
                authors TEXT[],
                summary TEXT,
                arxiv_id VARCHAR(20) UNIQUE,
                published_date TIMESTAMP,
                pdf_url TEXT,
                content TEXT,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        conn.commit()
        cur.close()
        conn.close()
        print("ScholarStream Database Initialized!")
    except Exception as e:
        print(f"Initialization Failed: {e}")

if __name__ == "__main__":
    init_db()
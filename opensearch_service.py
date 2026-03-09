from opensearchpy import OpenSearch
import os

class OpenSearchService:
    def __init__(self):
        self.host = os.getenv("OPENSEARCH_URL", "http://opensearch:9200")
        self.index_name = "arxiv-papers"
        self.client = OpenSearch(
            hosts=[self.host],
            http_compress=True,
            use_ssl=False,
            verify_certs=False,
            ssl_show_warn=False
        )

    def create_index(self):
        """Creates index with BM25 (text) and KNN (vector) search capabilities."""
        settings = {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": "100"
                }
            },
            "mappings": {
                "properties": {
                    "title": {"type": "text"},
                    "authors": {"type": "keyword"},
                    "summary": {"type": "text"},
                    "content": {"type": "text"}, # Full Markdown
                    "arxiv_id": {"type": "keyword"},
                    "published_date": {"type": "date"},
                    # 768 dimensions for nomic-embed-text
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": 768,
                        "method": {
                            "name": "hnsw",
                            "space_type": "l2",
                            "engine": "nmslib",
                            "parameters": {
                                "ef_construction": 128,
                                "m": 24
                            }
                        }
                    }
                }
            }
        }
        if not self.client.indices.exists(index=self.index_name):
            self.client.indices.create(index=self.index_name, body=settings)
            return True
        return False

    def index_chunk(self, chunk_id, paper_metadata, text, embedding):
        """Indexes an individual semantic chunk of a paper."""
        doc = {
            **paper_metadata,
            "content": text,
            "embedding": embedding,
            "chunk_id": chunk_id
        }
        return self.client.index(index=self.index_name, body=doc, id=chunk_id, refresh=True)
    
    def health_check(self):
        """Pings the OpenSearch cluster to verify connectivity."""
        try:
            return self.client.ping()
        except Exception as e:
            print(f"OpenSearch Health Check Failed: {e}")
            return False
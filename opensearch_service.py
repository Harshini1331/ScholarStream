from opensearchpy import OpenSearch
import os

class OpenSearchService:
    def __init__(self):
        self.host = os.getenv("OPENSEARCH__HOST", "http://opensearch:9200")
        self.index_name = os.getenv("OPENSEARCH__INDEX_NAME", "arxiv-papers")
        
        # Connect to the containerized OpenSearch
        self.client = OpenSearch(
            hosts=[self.host],
            http_compress=True,
            use_ssl=False,
            verify_certs=False,
            ssl_show_warn=False,
        )

    def health_check(self):
        try:
            return self.client.ping()
        except Exception as e:
            print(f"Connection Error: {e}")
            return False

    def create_index(self):
        """Creates the BM25 optimized index with proper mappings"""
        settings = {
            "settings": {
                "index": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0
                }
            },
            "mappings": {
                "properties": {
                    "title": {"type": "text", "analyzer": "standard"},
                    "authors": {"type": "keyword"},
                    "summary": {"type": "text"},
                    "content": {"type": "text"},
                    "arxiv_id": {"type": "keyword"},
                    "published_date": {"type": "date"},
                    "pdf_url": {"type": "keyword"}
                }
            }
        }
        
        if not self.client.indices.exists(index=self.index_name):
            self.client.indices.create(index=self.index_name, body=settings)
            return True
        return False

    def index_paper(self, paper_dict):
        """Uploads a single paper to the search index"""
        return self.client.index(
            index=self.index_name,
            body=paper_dict,
            id=paper_dict["arxiv_id"],
            refresh=True
        )
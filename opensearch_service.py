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
        """Creates index with BM25 (text) + KNN (vector) search capabilities.
        Uses English analyzer for improved BM25 relevance (stemming, stopwords).
        """
        settings = {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": "100"
                },
                "analysis": {
                    "analyzer": {
                        "english_analyzer": {
                            "type": "standard",
                            "stopwords": "_english_"
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "title": {
                        "type": "text",
                        "analyzer": "english_analyzer",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "authors": {"type": "keyword"},
                    "summary": {
                        "type": "text",
                        "analyzer": "english_analyzer"
                    },
                    "content": {
                        "type": "text",
                        "analyzer": "english_analyzer"
                    },
                    "section": {
                        "type": "text",
                        "fields": {
                            "keyword": {"type": "keyword", "ignore_above": 256}
                        }
                    },
                    "arxiv_id": {"type": "keyword"},
                    "published_date": {"type": "date"},
                    "pdf_url": {
                        "type": "text",
                        "fields": {
                            "keyword": {"type": "keyword", "ignore_above": 256}
                        }
                    },
                    "chunk_id": {
                        "type": "text",
                        "fields": {
                            "keyword": {"type": "keyword", "ignore_above": 256}
                        }
                    },
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

    def bm25_search(self, query: str, size: int = 10, from_: int = 0, arxiv_id_filter: str = None):
        """
        BM25 multi-field search across title (3x), summary (2x), content (1x).
        Supports:
          - Short queries (AI, ML, NN, CV) via match query
          - Fuzzy matching for typo tolerance
          - Highlighting of matched terms
          - Pagination via size/from
          - Optional arxiv_id filter
        """
        # Build the core multi-match query with field boosting
        must_clause = {
            "multi_match": {
                "query": query,
                "fields": [
                    "title^3",    # Title matches weighted 3x
                    "summary^2",  # Summary matches weighted 2x
                    "content^1"   # Content matches weighted 1x
                ],
                "type": "best_fields",
                "fuzziness": "AUTO",      # Handles typos automatically
                "minimum_should_match": "1"
            }
        }

        # Build optional filter
        search_body = {
            "size": size,
            "from": from_,
            "query": {
                "bool": {
                    "must": [must_clause]
                }
            },
            # Highlight matched terms in results
            "highlight": {
                "fields": {
                    "title": {
                        "pre_tags": ["<mark>"],
                        "post_tags": ["</mark>"],
                        "number_of_fragments": 1
                    },
                    "summary": {
                        "pre_tags": ["<mark>"],
                        "post_tags": ["</mark>"],
                        "number_of_fragments": 2,
                        "fragment_size": 200
                    },
                    "content": {
                        "pre_tags": ["<mark>"],
                        "post_tags": ["</mark>"],
                        "number_of_fragments": 2,
                        "fragment_size": 200
                    }
                }
            },
            # Return BM25 score explanation
            "_source": ["title", "arxiv_id", "summary", "section", "pdf_url", "authors", "published_date"]
        }

        # Apply optional arxiv_id filter
        if arxiv_id_filter:
            search_body["query"]["bool"]["filter"] = [
                {"term": {"arxiv_id": arxiv_id_filter}}
            ]

        response = self.client.search(index=self.index_name, body=search_body)

        # Format results
        results = []
        for hit in response["hits"]["hits"]:
            result = {
                "arxiv_id": hit["_source"].get("arxiv_id"),
                "title": hit["_source"].get("title"),
                "summary": hit["_source"].get("summary"),
                "section": hit["_source"].get("section"),
                "pdf_url": hit["_source"].get("pdf_url"),
                "authors": hit["_source"].get("authors"),
                "published_date": hit["_source"].get("published_date"),
                "bm25_score": hit["_score"],
                "highlights": hit.get("highlight", {})
            }
            results.append(result)

        return {
            "total": response["hits"]["total"]["value"],
            "results": results
        }

    def index_chunk(self, chunk_id, paper_metadata, text, embedding):
        """Indexes an individual semantic chunk of a paper."""
        doc = {
            **paper_metadata,
            "content": text,
            "embedding": embedding,
            "chunk_id": chunk_id
        }
        return self.client.index(
            index=self.index_name,
            body=doc,
            id=chunk_id,
            refresh=True
        )

    def health_check(self):
        """Pings the OpenSearch cluster to verify connectivity."""
        try:
            return self.client.ping()
        except Exception as e:
            print(f"OpenSearch Health Check Failed: {e}")
            return False

    def get_index_stats(self):
        """Returns document count and index size stats."""
        try:
            stats = self.client.indices.stats(index=self.index_name)
            count = self.client.count(index=self.index_name)
            return {
                "document_count": count["count"],
                "index_size_bytes": stats["indices"][self.index_name]["total"]["store"]["size_in_bytes"]
            }
        except Exception as e:
            return {"error": str(e)}
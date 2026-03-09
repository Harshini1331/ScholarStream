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
        """BM25 multi-field search with boosting, fuzzy matching, highlighting, pagination."""
        must_clause = {
            "multi_match": {
                "query": query,
                "fields": ["title^3", "summary^2", "content^1"],
                "type": "best_fields",
                "fuzziness": "AUTO",
                "minimum_should_match": "1"
            }
        }

        search_body = {
            "size": size,
            "from": from_,
            "query": {"bool": {"must": [must_clause]}},
            "highlight": {
                "fields": {
                    "title": {
                        "pre_tags": ["<mark>"], "post_tags": ["</mark>"],
                        "number_of_fragments": 1
                    },
                    "summary": {
                        "pre_tags": ["<mark>"], "post_tags": ["</mark>"],
                        "number_of_fragments": 2, "fragment_size": 200
                    },
                    "content": {
                        "pre_tags": ["<mark>"], "post_tags": ["</mark>"],
                        "number_of_fragments": 2, "fragment_size": 200
                    }
                }
            },
            "_source": ["title", "arxiv_id", "summary", "section", "pdf_url", "authors", "published_date"]
        }

        if arxiv_id_filter:
            search_body["query"]["bool"]["filter"] = [
                {"term": {"arxiv_id": arxiv_id_filter}}
            ]

        response = self.client.search(index=self.index_name, body=search_body)

        results = []
        for hit in response["hits"]["hits"]:
            results.append({
                "arxiv_id": hit["_source"].get("arxiv_id"),
                "title": hit["_source"].get("title"),
                "summary": hit["_source"].get("summary"),
                "section": hit["_source"].get("section"),
                "pdf_url": hit["_source"].get("pdf_url"),
                "authors": hit["_source"].get("authors"),
                "published_date": hit["_source"].get("published_date"),
                "bm25_score": hit["_score"],
                "highlights": hit.get("highlight", {}),
                "search_mode": "bm25"
            })

        return {
            "total": response["hits"]["total"]["value"],
            "results": results,
            "search_mode": "bm25"
        }

    def hybrid_search(self, query: str, query_vector: list, size: int = 10, rrf_k: int = 60):
        """
        Manual RRF (Reciprocal Rank Fusion) hybrid search for OpenSearch 2.11.0.

        Algorithm:
          1. Run BM25 search → ranked list A
          2. Run KNN vector search → ranked list B
          3. For each doc: rrf_score = 1/(rrf_k + rank_A) + 1/(rrf_k + rank_B)
          4. Re-rank by combined rrf_score and return top `size`

        rrf_k=60 is the standard constant from the original RRF paper.
        Docs appearing in BOTH lists get the highest scores.
        """
        fetch_size = max(size * 3, 30)

        # Step 1: BM25 search
        bm25_body = {
            "size": fetch_size,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["title^3", "summary^2", "content^1"],
                    "type": "best_fields",
                    "fuzziness": "AUTO"
                }
            },
            "_source": ["title", "arxiv_id", "summary", "section", "pdf_url", "authors", "published_date"]
        }
        bm25_response = self.client.search(index=self.index_name, body=bm25_body)

        # Step 2: KNN vector search
        knn_body = {
            "size": fetch_size,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_vector,
                        "k": fetch_size
                    }
                }
            },
            "_source": ["title", "arxiv_id", "summary", "section", "pdf_url", "authors", "published_date"]
        }
        knn_response = self.client.search(index=self.index_name, body=knn_body)

        # Step 3: Build rank maps {doc_id: rank} (rank starts at 1)
        bm25_ranks = {
            hit["_id"]: rank + 1
            for rank, hit in enumerate(bm25_response["hits"]["hits"])
        }
        knn_ranks = {
            hit["_id"]: rank + 1
            for rank, hit in enumerate(knn_response["hits"]["hits"])
        }

        # Step 4: Collect all unique docs from both result sets
        all_docs = {}
        for hit in bm25_response["hits"]["hits"]:
            all_docs[hit["_id"]] = hit["_source"]
        for hit in knn_response["hits"]["hits"]:
            if hit["_id"] not in all_docs:
                all_docs[hit["_id"]] = hit["_source"]

        # Step 5: Compute RRF score for each doc
        rrf_scores = {}
        for doc_id in all_docs:
            score = 0.0
            if doc_id in bm25_ranks:
                score += 1.0 / (rrf_k + bm25_ranks[doc_id])
            if doc_id in knn_ranks:
                score += 1.0 / (rrf_k + knn_ranks[doc_id])
            rrf_scores[doc_id] = score

        # Step 6: Sort by RRF score descending, return top `size`
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:size]

        results = []
        for doc_id, rrf_score in ranked:
            src = all_docs[doc_id]
            results.append({
                "arxiv_id": src.get("arxiv_id"),
                "title": src.get("title"),
                "summary": src.get("summary"),
                "section": src.get("section"),
                "pdf_url": src.get("pdf_url"),
                "authors": src.get("authors"),
                "published_date": src.get("published_date"),
                "rrf_score": round(rrf_score, 6),
                "bm25_rank": bm25_ranks.get(doc_id),
                "knn_rank": knn_ranks.get(doc_id),
                "search_mode": "hybrid"
            })

        return {
            "total": len(all_docs),
            "results": results,
            "search_mode": "hybrid"
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
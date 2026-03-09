"""
Langfuse observability service for ScholarStream.
Uses Langfuse v3 SDK with start_as_current_span context managers.
"""

import os
from langfuse import Langfuse

SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
BASE_URL = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")


class TracingService:
    def __init__(self):
        if SECRET_KEY and PUBLIC_KEY:
            try:
                self.client = Langfuse(
                    secret_key=SECRET_KEY,
                    public_key=PUBLIC_KEY,
                    host=BASE_URL,
                )
                self.enabled = True
                print("Langfuse tracing enabled")
            except Exception as e:
                self.client = None
                self.enabled = False
                print(f"Langfuse init failed — tracing disabled: {e}")
        else:
            self.client = None
            self.enabled = False
            print("Langfuse keys not set — tracing disabled")

    def trace_ask(self, question, answer, sources, top_k, use_hybrid,
                  cache_hit, response_time_s, embed_time_s=0,
                  retrieve_time_s=0, generate_time_s=0):
        if not self.enabled:
            return
        try:
            with self.client.start_as_current_span(
                name="rag-ask",
                input={"question": question, "top_k": top_k, "use_hybrid": use_hybrid},
            ):
                self.client.update_current_trace(
                    output={"answer": answer, "sources": sources},
                    metadata={"cache_hit": cache_hit, "response_time_s": response_time_s,
                              "search_mode": "hybrid" if use_hybrid else "vector"},
                    tags=["ask", "cache-hit" if cache_hit else "cache-miss"],
                )
                if not cache_hit:
                    with self.client.start_as_current_span(
                        name="embed-query", metadata={"duration_s": embed_time_s}
                    ):
                        pass
                    with self.client.start_as_current_span(
                        name="retrieve-context",
                        metadata={"duration_s": retrieve_time_s, "top_k": top_k}
                    ):
                        pass
                    with self.client.start_as_current_generation(
                        name="llm-generate", model="llama3",
                        input=question, output=answer,
                        metadata={"duration_s": generate_time_s}
                    ):
                        pass
            self.client.flush()
        except Exception as e:
            print(f"Langfuse trace error (non-fatal): {e}")

    def trace_stream(self, question, answer, sources, top_k, use_hybrid,
                     cache_hit, response_time_s):
        if not self.enabled:
            return
        try:
            with self.client.start_as_current_span(
                name="rag-stream",
                input={"question": question, "top_k": top_k, "use_hybrid": use_hybrid},
            ):
                self.client.update_current_trace(
                    output={"answer": answer[:500], "sources": sources},
                    metadata={"cache_hit": cache_hit, "response_time_s": response_time_s,
                              "search_mode": "hybrid" if use_hybrid else "vector"},
                    tags=["stream", "cache-hit" if cache_hit else "cache-miss"],
                )
            self.client.flush()
        except Exception as e:
            print(f"Langfuse stream trace error (non-fatal): {e}")
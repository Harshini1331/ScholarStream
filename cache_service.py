"""
Redis caching service for ScholarStream.
Cache key: sha256(question + top_k + use_hybrid)
TTL: 24 hours by default (configurable via REDIS__TTL_HOURS)
"""

import os
import json
import hashlib
import redis


REDIS_HOST = os.getenv("REDIS__HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS__PORT", "6379"))
TTL_SECONDS = int(os.getenv("REDIS__TTL_HOURS", "24")) * 3600
KEY_PREFIX = "scholarstream:ask:"


class CacheService:
    def __init__(self):
        try:
            self.client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                decode_responses=True,
                socket_connect_timeout=2,
            )
            self.client.ping()
            self.enabled = True
            print(f"Redis connected at {REDIS_HOST}:{REDIS_PORT}")
        except Exception as e:
            self.client = None
            self.enabled = False
            print(f"Redis unavailable — caching disabled: {e}")

    def _key(self, question: str, top_k: int, use_hybrid: bool) -> str:
        raw = f"{question.strip().lower()}|{top_k}|{use_hybrid}"
        digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
        return f"{KEY_PREFIX}{digest}"

    def get(self, question: str, top_k: int, use_hybrid: bool) -> dict | None:
        if not self.enabled:
            return None
        try:
            value = self.client.get(self._key(question, top_k, use_hybrid))
            return json.loads(value) if value else None
        except Exception:
            return None

    def set(self, question: str, top_k: int, use_hybrid: bool, data: dict) -> bool:
        if not self.enabled:
            return False
        try:
            self.client.setex(
                self._key(question, top_k, use_hybrid),
                TTL_SECONDS,
                json.dumps(data)
            )
            return True
        except Exception:
            return False

    def flush_all(self) -> int:
        """Delete all ScholarStream cache keys. Returns count deleted."""
        if not self.enabled:
            return 0
        try:
            keys = self.client.keys(f"{KEY_PREFIX}*")
            if keys:
                return self.client.delete(*keys)
            return 0
        except Exception:
            return 0

    def stats(self) -> dict:
        if not self.enabled:
            return {"enabled": False}
        try:
            keys = self.client.keys(f"{KEY_PREFIX}*")
            info = self.client.info("memory")
            return {
                "enabled": True,
                "cached_responses": len(keys),
                "ttl_hours": TTL_SECONDS // 3600,
                "used_memory": info.get("used_memory_human", "N/A"),
            }
        except Exception:
            return {"enabled": True, "error": "Could not fetch stats"}
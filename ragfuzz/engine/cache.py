"""Deterministic caching for request deduplication."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CacheEntry:
    """A cache entry."""

    key: str
    value: str
    timestamp: float
    provider_id: str
    model_id: str
    temperature: float
    max_tokens: int | None


class Cache:
    """Deterministic cache for LLM requests."""

    def __init__(self, cache_dir: str | Path = ".cache"):
        """Initialize cache.

        Args:
            cache_dir: Directory to store cache database.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "ragfuzz_cache.db"

        self._init_db()

    def __del__(self):
        """Ensure database connection is closed."""
        self.close()

    def _init_db(self) -> None:
        """Initialize the SQLite cache database."""
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value TEXT,
                timestamp REAL,
                provider_id TEXT,
                model_id TEXT,
                temperature REAL,
                max_tokens INTEGER
            )
        """
        )

        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_key ON cache(key)
        """
        )

        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_timestamp ON cache(timestamp)
        """
        )

        self.conn.commit()

    def generate_key(
        self,
        provider_id: str,
        model_id: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int | None,
        mutator_node_id: str | None = None,
    ) -> str:
        """Generate a cache key for a request.

        Args:
            provider_id: The provider ID.
            model_id: The model ID.
            messages: The messages array.
            temperature: The sampling temperature.
            max_tokens: Maximum tokens.
            mutator_node_id: Optional mutator node ID.

        Returns:
            A cache key hash.
        """
        data = {
            "provider_id": provider_id,
            "model_id": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "mutator_node_id": mutator_node_id,
        }

        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def get(
        self,
        key: str,
    ) -> CacheEntry | None:
        """Get a cached value.

        Args:
            key: The cache key.

        Returns:
            CacheEntry if found, None otherwise.
        """
        cursor = self.conn.execute(
            "SELECT key, value, timestamp, provider_id, model_id, temperature, max_tokens FROM cache WHERE key = ?",
            (key,),
        )

        row = cursor.fetchone()

        if not row:
            return None

        return CacheEntry(
            key=row[0],
            value=row[1],
            timestamp=row[2],
            provider_id=row[3],
            model_id=row[4],
            temperature=row[5],
            max_tokens=row[6],
        )

    def set(
        self,
        key: str,
        value: str,
        provider_id: str,
        model_id: str,
        temperature: float,
        max_tokens: int | None,
    ) -> None:
        """Store a value in the cache.

        Args:
            key: The cache key.
            value: The value to cache.
            provider_id: The provider ID.
            model_id: The model ID.
            temperature: The sampling temperature.
            max_tokens: Maximum tokens.
        """
        import time

        timestamp = time.time()

        self.conn.execute(
            """
            INSERT OR REPLACE INTO cache
            (key, value, timestamp, provider_id, model_id, temperature, max_tokens)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (key, value, timestamp, provider_id, model_id, temperature, max_tokens),
        )

        self.conn.commit()

    def clear(self) -> None:
        """Clear all cache entries."""
        self.conn.execute("DELETE FROM cache")
        self.conn.commit()

    def cleanup_old(self, max_age_seconds: int = 86400) -> None:
        """Remove cache entries older than max_age_seconds.

        Args:
            max_age_seconds: Maximum age in seconds (default: 24 hours).
        """
        import time

        cutoff = time.time() - max_age_seconds

        self.conn.execute(
            "DELETE FROM cache WHERE timestamp < ?",
            (cutoff,),
        )

        self.conn.commit()

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats.
        """
        cursor = self.conn.execute("SELECT COUNT(*) FROM cache")
        total_entries = cursor.fetchone()[0]

        cursor = self.conn.execute("SELECT AVG(timestamp) FROM cache WHERE timestamp > 0")
        avg_timestamp = cursor.fetchone()[0]

        import time

        avg_age = (time.time() - avg_timestamp) if avg_timestamp else 0

        return {
            "total_entries": total_entries,
            "avg_age_seconds": avg_age,
            "db_path": str(self.db_path),
        }

    def close(self) -> None:
        """Close the database connection."""
        if hasattr(self, "conn") and self.conn:
            self.conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False

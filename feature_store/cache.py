"""
feature_store/cache.py
───────────────────────
In-Memory LRU Cache Layer for QuantSphereX Feature Store.
Accelerates training and inference data loading by holding recent feature panel slices in memory.
"""

from __future__ import annotations
import logging
from collections import OrderedDict
from typing import Optional, Tuple, Any
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureStoreCache:
    """LRU In-Memory Cache for Feature Store DataFrames."""

    def __init__(self, max_entries: int = 4):
        self.max_entries = max_entries
        self._cache: OrderedDict[Tuple[str, str, str, str], pd.DataFrame] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _make_key(self, version_id: str, tickers: Tuple[str, ...], start_date: str, end_date: str) -> Tuple[str, str, str, str]:
        tickers_str = ",".join(sorted(tickers))
        return (version_id, tickers_str, start_date, end_date)

    def get(self, version_id: str, tickers: Tuple[str, ...], start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Retrieves DataFrame from LRU cache if hit; updates access order."""
        key = self._make_key(version_id, tickers, start_date, end_date)
        if key in self._cache:
            self.hits += 1
            self._cache.move_to_end(key)
            logger.debug(f"[FeatureStoreCache] CACHE HIT for version={version_id}")
            return self._cache[key].copy()
        self.misses += 1
        return None

    def put(self, version_id: str, tickers: Tuple[str, ...], start_date: str, end_date: str, df: pd.DataFrame) -> None:
        """Stores DataFrame in LRU cache, evicting oldest entry if over capacity."""
        if df.empty:
            return
        key = self._make_key(version_id, tickers, start_date, end_date)
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = df.copy()
        if len(self._cache) > self.max_entries:
            evicted_key, _ = self._cache.popitem(last=False)
            logger.debug(f"[FeatureStoreCache] EVICTED oldest cache entry '{evicted_key[0]}'")

    def clear(self) -> None:
        """Flushes in-memory cache."""
        self._cache.clear()
        self.hits = 0
        self.misses = 0

    def stats(self) -> dict:
        total = self.hits + self.misses
        hit_rate = (self.hits / total) if total > 0 else 0.0
        return {
            "entries": len(self._cache),
            "max_entries": self.max_entries,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate
        }

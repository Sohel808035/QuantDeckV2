"""
feature_store/reader.py
────────────────────────
Feature Store Reader & Query Engine for QuantSphereX Feature Store.
Provides point-in-time inference slicing, walk-forward training batch filtering, and LRU cache retrieval.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional, List, Tuple
import pandas as pd

from feature_store.schema import FeatureVersion
from feature_store.registry import FeatureRegistry, DEFAULT_FEATURE_STORE_DIR
from feature_store.cache import FeatureStoreCache

logger = logging.getLogger(__name__)


class FeatureStoreReader:
    """Provides high-performance reads of versioned feature panels from Parquet storage and LRU memory cache."""

    def __init__(
        self,
        registry: Optional[FeatureRegistry] = None,
        cache: Optional[FeatureStoreCache] = None,
        store_dir: Path = DEFAULT_FEATURE_STORE_DIR
    ):
        self.store_dir = Path(store_dir)
        self.registry = registry or FeatureRegistry(store_dir=self.store_dir)
        self.cache = cache or FeatureStoreCache(max_entries=4)

    def load_training_set(
        self,
        version_id: Optional[str] = None,
        tickers: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        schema_id: str = "quantdeck_alpha_v5"
    ) -> pd.DataFrame:
        """
        Loads a feature panel for walk-forward training.

        Args:
            version_id: Version ID to read. If None, auto-resolves latest version for schema_id.
            tickers: Optional list of tickers to filter.
            start_date: Optional start date filter (YYYY-MM-DD).
            end_date: Optional end date filter (YYYY-MM-DD).
            schema_id: Schema ID used for auto-resolution if version_id is None.

        Returns:
            Filtered MultiIndex (Date, Ticker) feature panel DataFrame.
        """
        # Auto-resolve latest version if not specified
        if not version_id:
            latest_manifest = self.registry.get_latest_version(schema_id=schema_id)
            if not latest_manifest:
                raise FileNotFoundError(f"[FeatureStoreReader] No registered feature versions for schema '{schema_id}'.")
            version_id = latest_manifest.version_id

        tickers_tuple = tuple(sorted(tickers)) if tickers else ("ALL",)
        s_date = start_date or "MIN"
        e_date = end_date or "MAX"

        # 1. Check LRU Cache
        cached_df = self.cache.get(version_id, tickers_tuple, s_date, e_date)
        if cached_df is not None:
            return cached_df

        # 2. Read Parquet File
        parquet_path = self.store_dir / f"features_{version_id}.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"[FeatureStoreReader] Parquet file not found: {parquet_path}")

        df = pd.read_parquet(parquet_path)

        # 3. Apply Filters
        if isinstance(df.index, pd.MultiIndex):
            # Sort index for fast slicing
            if not df.index.is_monotonic_increasing:
                df = df.sort_index()

            # Date filtering
            if start_date or end_date:
                idx_dates = df.index.get_level_values("Date")
                cond = pd.Series(True, index=df.index)
                if start_date:
                    cond &= (idx_dates >= pd.Timestamp(start_date))
                if end_date:
                    cond &= (idx_dates <= pd.Timestamp(end_date))
                df = df[cond]

            # Ticker filtering
            if tickers and "Ticker" in df.index.names:
                df = df.xs(slice(None), level="Date", drop_level=False)
                df = df[df.index.get_level_values("Ticker").isin(tickers)]

        logger.info(
            f"[FeatureStoreReader] Loaded training set '{version_id}': "
            f"{len(df):,} rows | {len(df.columns)} features"
        )

        # 4. Store in LRU Cache
        self.cache.put(version_id, tickers_tuple, s_date, e_date, df)
        return df

    def load_inference_set(
        self,
        date: str,
        tickers: List[str],
        version_id: Optional[str] = None,
        schema_id: str = "quantdeck_alpha_v5"
    ) -> pd.DataFrame:
        """
        Loads point-in-time cross-sectional features for live trading inference.

        Args:
            date: Target prediction date string (YYYY-MM-DD).
            tickers: Tradable universe tickers.
            version_id: Optional feature dataset version. Auto-resolves latest if None.
            schema_id: Schema ID for auto-resolution.

        Returns:
            Cross-sectional DataFrame indexed by Ticker for the target date.
        """
        target_date = pd.Timestamp(date)
        
        # Read window containing target_date
        panel = self.load_training_set(
            version_id=version_id,
            tickers=tickers,
            start_date=date,
            end_date=date,
            schema_id=schema_id
        )

        if panel.empty:
            logger.warning(f"[FeatureStoreReader] No features found for inference on date {date}")
            return pd.DataFrame()

        # Slice target date level
        try:
            if isinstance(panel.index, pd.MultiIndex):
                cs = panel.xs(target_date, level="Date")
            else:
                cs = panel
            
            # Reindex to requested universe tickers
            cs = cs.reindex([t for t in tickers if t in cs.index])
            logger.info(f"[FeatureStoreReader] Point-in-time inference slice for {date}: {len(cs)} tickers")
            return cs
        except KeyError:
            logger.warning(f"[FeatureStoreReader] Date {date} not present in feature panel.")
            return pd.DataFrame()

    def load_latest(
        self,
        tickers: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        schema_id: str = "quantdeck_alpha_v5"
    ) -> pd.DataFrame:
        """Helper to auto-resolve latest registered version and load DataFrame."""
        return self.load_training_set(
            version_id=None,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            schema_id=schema_id
        )

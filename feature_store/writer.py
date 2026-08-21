"""
feature_store/writer.py
────────────────────────
Atomic Feature Store Writer for QuantSphereX Feature Store.
Validates, hashes, tracks lineage, and atomically persists feature panels with JSON manifests.
"""

from __future__ import annotations
import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd

from feature_store.schema import FeatureSchema, FeatureVersion, ValidationReport
from feature_store.registry import FeatureRegistry, DEFAULT_FEATURE_STORE_DIR
from feature_store.validator import FeatureStoreValidator
from feature_store.lineage import LineageTracker

logger = logging.getLogger(__name__)


class FeatureStoreWriter:
    """Handles atomic writing of feature panels to Parquet format with version registration."""

    def __init__(
        self,
        registry: Optional[FeatureRegistry] = None,
        lineage_tracker: Optional[LineageTracker] = None,
        store_dir: Path = DEFAULT_FEATURE_STORE_DIR
    ):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.registry = registry or FeatureRegistry(store_dir=self.store_dir)
        self.lineage_tracker = lineage_tracker or LineageTracker(lineage_path=self.store_dir / "lineage.jsonl")

    def _compute_data_hash(self, df: pd.DataFrame) -> str:
        """Computes deterministic SHA-256 content hash of feature DataFrame."""
        sample_bytes = pd.util.hash_pandas_object(df, index=True).values.tobytes()
        return hashlib.sha256(sample_bytes).hexdigest()[:16]

    def _compute_tickers_hash(self, df: pd.DataFrame) -> str:
        """Computes hash of sorted tickers contained in MultiIndex."""
        if isinstance(df.index, pd.MultiIndex) and "Ticker" in df.index.names:
            tickers = sorted(df.index.get_level_values("Ticker").unique().tolist())
        else:
            tickers = ["ALL"]
        return hashlib.md5(",".join(tickers).encode("utf-8")).hexdigest()[:8]

    def write(
        self,
        panel: pd.DataFrame,
        schema: Optional[FeatureSchema] = None,
        version_id: Optional[str] = None,
        input_dataset_name: str = "stock_panel",
        input_dataset_version: str = "v1.0",
        transform_name: str = "FeatureFactoryPipeline",
        transform_config: Optional[Dict[str, Any]] = None,
        overwrite: bool = False
    ) -> FeatureVersion:
        """
        Validates, atomically persists feature panel to Parquet, and updates Feature Registry.

        Args:
            panel: Feature panel DataFrame (Date/Ticker MultiIndex).
            schema: Optional FeatureSchema (uses default alpha schema if None).
            version_id: Optional custom version ID string (auto-generated if None).
            input_dataset_name: Name of input dataset.
            input_dataset_version: Version of input dataset.
            transform_name: Name of transformation applied.
            transform_config: Configuration dictionary of feature pipeline.
            overwrite: Whether to overwrite existing file if version_id collision occurs.

        Returns:
            FeatureVersion manifest detailing persisted dataset.
        """
        if panel.empty:
            raise ValueError("[FeatureStoreWriter] Cannot write empty feature panel DataFrame.")

        # 1. Resolve & Register Schema
        if schema is None:
            active_cols = [c for c in panel.columns if c not in ["Open", "High", "Low", "Close", "Volume"]]
            schema = FeatureRegistry.create_default_schema(feature_cols=active_cols)
        self.registry.register_schema(schema)

        # 2. Run Data Quality Audit
        validator = FeatureStoreValidator(schema)
        report: ValidationReport = validator.validate(panel)
        if not report.is_valid:
            logger.error(f"[FeatureStoreWriter] Feature panel validation failed: {report.feature_errors}")
            raise ValueError(f"[FeatureStoreWriter] Data quality audit failed with errors: {report.feature_errors}")

        # 3. Derive Metadata & Dates
        if isinstance(panel.index, pd.MultiIndex) and "Date" in panel.index.names:
            dates = panel.index.get_level_values("Date").unique().sort_values()
            start_date = dates[0].strftime("%Y-%m-%d")
            end_date = dates[-1].strftime("%Y-%m-%d")
        else:
            start_date = "1970-01-01"
            end_date = "2099-12-31"

        data_hash = self._compute_data_hash(panel)
        tickers_hash = self._compute_tickers_hash(panel)

        if not version_id:
            today_str = pd.Timestamp.now().strftime("%Y%m%d")
            version_id = f"feat_{schema.schema_id}_v{today_str}_{data_hash[:8]}"

        target_file = self.store_dir / f"features_{version_id}.parquet"
        if target_file.exists() and not overwrite:
            logger.info(f"[FeatureStoreWriter] Version '{version_id}' already exists at {target_file}")
            existing_ver = self.registry.get_version(version_id)
            if existing_ver:
                return existing_ver

        # 4. Atomic Write (Temp File -> Replace)
        temp_file = None
        try:
            with tempfile.NamedTemporaryFile("wb", dir=self.store_dir, delete=False, suffix=".parquet") as tf:
                panel.to_parquet(tf.name, compression="snappy")
                temp_file = Path(tf.name)
            os.replace(temp_file, target_file)
        except Exception as exc:
            if temp_file and temp_file.exists():
                try:
                    os.unlink(temp_file)
                except Exception:
                    pass
            logger.critical(f"[FeatureStoreWriter] Failed to write Parquet file atomically: {exc}")
            raise

        # 5. Record Lineage
        lineage_id = self.lineage_tracker.record_lineage(
            input_dataset_name=input_dataset_name,
            input_dataset_version=input_dataset_version,
            output_feature_version=version_id,
            transform_name=transform_name,
            transform_config=transform_config
        )

        # 6. Register Manifest
        feat_cols = [c for c in panel.columns if c in schema.features]
        manifest = FeatureVersion(
            version_id=version_id,
            schema_id=schema.schema_id,
            start_date=start_date,
            end_date=end_date,
            tickers_hash=tickers_hash,
            data_hash=data_hash,
            row_count=len(panel),
            feature_count=len(feat_cols),
            lineage_id=lineage_id
        )
        self.registry.register_version(manifest)

        logger.info(f"[FeatureStoreWriter] Atomically wrote {len(panel):,} rows to {target_file}")
        return manifest

"""
feature_store/lineage.py
────────────────────────
Feature Transformation Lineage Tracker for QuantSphereX Feature Store.
Records transformation graph, input dataset dependencies, and recipe parameters.
"""

from __future__ import annotations
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

DEFAULT_LINEAGE_PATH = Path("data_cache/feature_store/lineage.jsonl")


class LineageTracker:
    """Tracks transformation graph from raw/cached data to versioned feature store panels."""

    def __init__(self, lineage_path: Path = DEFAULT_LINEAGE_PATH):
        self.lineage_path = Path(lineage_path)
        self.lineage_path.parent.mkdir(parents=True, exist_ok=True)

    def record_lineage(
        self,
        input_dataset_name: str,
        input_dataset_version: str,
        output_feature_version: str,
        transform_name: str,
        transform_config: Optional[Dict[str, Any]] = None,
        notes: str = ""
    ) -> str:
        """Records a transformation event linking input dataset version to output feature version."""
        lineage_id = str(uuid.uuid4())[:12]
        record = {
            "lineage_id": lineage_id,
            "input_dataset_name": input_dataset_name,
            "input_dataset_version": input_dataset_version,
            "output_feature_version": output_feature_version,
            "transform_name": transform_name,
            "transform_config": transform_config or {},
            "notes": notes,
            "created_at": datetime.utcnow().isoformat()
        }

        with open(self.lineage_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        logger.info(f"[LineageTracker] Recorded lineage '{lineage_id}': {input_dataset_name} -> {output_feature_version}")
        return lineage_id

    def get_lineage(self, lineage_id: str) -> Optional[Dict[str, Any]]:
        """Find line by lineage_id."""
        if not self.lineage_path.exists():
            return None
        with open(self.lineage_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if rec.get("lineage_id") == lineage_id:
                        return rec
        return None

    def get_upstream(self, feature_version: str) -> List[Dict[str, Any]]:
        """Find all upstream input datasets that contributed to feature_version."""
        if not self.lineage_path.exists():
            return []
        matches = []
        with open(self.lineage_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if rec.get("output_feature_version") == feature_version:
                        matches.append(rec)
        return matches

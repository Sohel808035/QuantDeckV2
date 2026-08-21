"""
feature_store/registry.py
──────────────────────────
Feature Catalog and Metadata Version Manager for QuantSphereX Feature Store.
Manages schema registration, dataset version manifests, and metadata catalog queries.
"""

from __future__ import annotations
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any

from feature_store.schema import FeatureSchema, FeatureDefinition, FeatureVersion

logger = logging.getLogger(__name__)

DEFAULT_FEATURE_STORE_DIR = Path("data_cache/feature_store")
DEFAULT_REGISTRY_PATH = DEFAULT_FEATURE_STORE_DIR / "registry.json"


class FeatureRegistry:
    """JSON-backed catalog of feature schemas and persisted feature dataset versions."""

    def __init__(
        self,
        registry_path: Path = DEFAULT_REGISTRY_PATH,
        store_dir: Path = DEFAULT_FEATURE_STORE_DIR
    ):
        self.registry_path = Path(registry_path)
        self.store_dir = Path(store_dir)
        self._ensure_paths()

    def _ensure_paths(self):
        self.store_dir.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            self._write_registry({"schemas": {}, "versions": {}})

    def _read_registry(self) -> Dict[str, Dict[str, Any]]:
        import time
        for attempt in range(5):
            try:
                if self.registry_path.exists():
                    with open(self.registry_path, "r", encoding="utf-8") as f:
                        return json.load(f)
                return {"schemas": {}, "versions": {}}
            except (PermissionError, OSError) as e:
                if attempt == 4:
                    logger.error(f"[FeatureRegistry] Failed to read registry JSON: {e}")
                    return {"schemas": {}, "versions": {}}
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"[FeatureRegistry] Failed to read registry JSON: {e}")
                return {"schemas": {}, "versions": {}}
        return {"schemas": {}, "versions": {}}

    def _write_registry(self, data: Dict[str, Dict[str, Any]]):
        temp_file = None
        try:
            parent = self.registry_path.parent
            parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile("w", dir=parent, delete=False, encoding="utf-8") as tf:
                json.dump(data, tf, indent=4)
                temp_file = Path(tf.name)

            replaced = False
            import time
            for _ in range(5):
                try:
                    if self.registry_path.exists():
                        try:
                            self.registry_path.unlink()
                        except (PermissionError, OSError):
                            pass
                    os.replace(temp_file, self.registry_path)
                    replaced = True
                    break
                except (PermissionError, OSError):
                    time.sleep(0.5)

            if not replaced:
                import shutil
                shutil.copyfile(str(temp_file), str(self.registry_path))
                try:
                    os.unlink(temp_file)
                except Exception:
                    pass
        except Exception as e:
            logger.critical(f"[FeatureRegistry] Failed to write registry atomically: {e}")
            if temp_file and temp_file.exists():
                try:
                    os.unlink(temp_file)
                except Exception:
                    pass
            raise

    # ── Schema Operations ──────────────────────────────────────────────────────

    def register_schema(self, schema: FeatureSchema) -> None:
        """Register or update a feature schema in the catalog."""
        reg = self._read_registry()
        key = f"{schema.schema_id}:{schema.version}"
        reg["schemas"][key] = schema.model_dump()
        self._write_registry(reg)
        logger.info(f"[FeatureRegistry] Registered schema '{key}' ({len(schema.features)} features)")

    def get_schema(self, schema_id: str, version: str = "1.0.0") -> Optional[FeatureSchema]:
        """Retrieve a specific FeatureSchema by ID and version."""
        reg = self._read_registry()
        key = f"{schema_id}:{version}"
        if key in reg.get("schemas", {}):
            return FeatureSchema(**reg["schemas"][key])
        return None

    # ── Dataset Version Operations ─────────────────────────────────────────────

    def register_version(self, version_info: FeatureVersion) -> None:
        """Register a new feature dataset version manifest."""
        reg = self._read_registry()
        reg["versions"][version_info.version_id] = version_info.model_dump()
        self._write_registry(reg)
        logger.info(f"[FeatureRegistry] Registered dataset version '{version_info.version_id}'")

    def get_version(self, version_id: str) -> Optional[FeatureVersion]:
        """Retrieve a FeatureVersion by version_id."""
        reg = self._read_registry()
        if version_id in reg.get("versions", {}):
            return FeatureVersion(**reg["versions"][version_id])
        return None

    def list_versions(self, schema_id: Optional[str] = None) -> List[FeatureVersion]:
        """List registered dataset versions, optionally filtered by schema_id."""
        reg = self._read_registry()
        versions = []
        for val in reg.get("versions", {}).values():
            ver = FeatureVersion(**val)
            if schema_id is None or ver.schema_id == schema_id:
                versions.append(ver)
        return versions

    def get_latest_version(self, schema_id: Optional[str] = None) -> Optional[FeatureVersion]:
        """Retrieve the most recently created dataset version."""
        versions = self.list_versions(schema_id=schema_id)
        if not versions:
            return None
        versions.sort(key=lambda v: v.created_at)
        return versions[-1]

    # ── Factory Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def create_default_schema(
        schema_id: str = "quantdeck_alpha_v5",
        version: str = "1.0.0",
        feature_cols: Optional[List[str]] = None,
    ) -> FeatureSchema:
        """Helper to construct the default QuantDeck alpha schema for active features."""
        from feature_layer.implementations import FEATURE_COLS, DESCENDING_FEATURES

        cols = feature_cols if feature_cols is not None else FEATURE_COLS
        feature_defs = {}
        for col in cols:
            is_descending = col in DESCENDING_FEATURES
            # Fundamental factors or dynamic features can have higher missing percentages
            is_fundamental = col in ["roe", "roa", "earnings_growth"]
            max_null = 1.0 if is_fundamental else 0.50

            feature_defs[col] = FeatureDefinition(
                name=col,
                dtype="float64",
                category="AlphaSignal",
                description=f"QuantDeck factor column (inverted rank={is_descending})",
                min_value=-10.0,
                max_value=10.0,
                allow_null=True,
                max_null_pct=max_null,
                is_lagged=True,
            )

        schema = FeatureSchema(
            schema_id=schema_id,
            version=version,
            features=feature_defs,
        )
        return schema

"""
alpha_layer/governance/registry.py
══════════════════════════════════
Model Registry and Metadata Manager for QuantSphereX Institutional.
Provides robust schema validation, serialization, thread-safe file I/O,
and automatic rollbacks.
"""

from __future__ import annotations
import os
import json
import logging
import pickle
import shutil
import tempfile
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_PATH = Path("data_cache/models/registry.json")
DEFAULT_MODELS_DIR = Path("data_cache/models")

class ModelMetadata(BaseModel):
    """Institutional schema for model governance metadata."""
    model_id: str = Field(..., description="Unique identifier for the model family")
    version: str = Field(..., description="Semantic version string, e.g., '1.0.0'")
    training_date: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    dataset_version: str = Field(..., description="Hash/descriptor of the training dataset version")
    feature_version: str = Field(..., description="Hash/descriptor of the feature engineering pipeline/cols")
    hyperparameters: Dict[str, Any] = Field(..., description="Hyperparameters used for training")
    validation_ic: float = Field(..., description="Validation Information Coefficient")
    train_ic: float = Field(..., description="Training Information Coefficient")
    sharpe: float = Field(..., description="Annualized Net Sharpe Ratio")
    drawdown: float = Field(..., description="Max Drawdown encountered in backtesting")
    status: str = Field("Experimental", description="Governance status: 'Production' or 'Experimental'")
    git_commit_hash: str = Field(..., description="Git commit hash at time of training")
    notes: Optional[str] = Field(None, description="Optional annotations or justification notes")

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        if v not in {"Production", "Experimental"}:
            raise ValueError("Status must be either 'Production' or 'Experimental'")
        return v

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        # Basic validation: ensure format like '1.2.3' or similar dot-separated numbers
        parts = v.split(".")
        if len(parts) < 2:
            raise ValueError("Version must be dot-separated, e.g., '1.0.0' or '1.0'")
        for p in parts:
            if not p.replace("v", "").isdigit():
                raise ValueError(f"Version components must be numeric: {v}")
        return v


class ModelRegistry:
    """Manages reading, writing, and querying the Model Registry database."""

    def __init__(self, registry_path: Path = DEFAULT_REGISTRY_PATH, models_dir: Path = DEFAULT_MODELS_DIR):
        self.registry_path = Path(registry_path)
        self.models_dir = Path(models_dir)
        self._ensure_paths()

    def _ensure_paths(self):
        self.models_dir.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            self._write_registry({})

    def _read_registry(self) -> Dict[str, Dict[str, Any]]:
        """Read registry metadata with automatic file lock safety (atomic-read emulation)."""
        import time
        for attempt in range(5):
            try:
                if self.registry_path.exists():
                    with open(self.registry_path, "r", encoding="utf-8") as f:
                        return json.load(f)
                return {}
            except (PermissionError, OSError) as e:
                if attempt == 4:
                    logger.error(f"Failed to read model registry JSON: {e}")
                    return {}
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Failed to read model registry JSON: {e}")
                return {}
        return {}

    def _write_registry(self, data: Dict[str, Dict[str, Any]]):
        """Write registry atomically via temp file replace to prevent corruption."""
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
            logger.critical(f"Failed to write model registry JSON atomically: {e}")
            if temp_file and temp_file.exists():
                try:
                    os.unlink(temp_file)
                except Exception:
                    pass
            raise

    def register_model(self, metadata: ModelMetadata) -> None:
        """Register a new model's metadata into the Model Registry."""
        registry = self._read_registry()
        key = f"{metadata.model_id}:{metadata.version}"
        registry[key] = metadata.model_dump()
        self._write_registry(registry)
        logger.info(f"Registered model {key} successfully in status: {metadata.status}")

    def get_model_metadata(self, model_id: str, version: str) -> Optional[ModelMetadata]:
        """Retrieve metadata for a specific model ID and version."""
        registry = self._read_registry()
        key = f"{model_id}:{version}"
        if key in registry:
            return ModelMetadata(**registry[key])
        return None

    def list_models(self, model_id: Optional[str] = None) -> List[ModelMetadata]:
        """List all models or filter by model_id."""
        registry = self._read_registry()
        models = []
        for key, val in registry.items():
            meta = ModelMetadata(**val)
            if model_id is None or meta.model_id == model_id:
                models.append(meta)
        return models

    def get_latest_production_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Get the latest registered 'Production' model for a given model_id based on version order."""
        models = self.list_models(model_id=model_id)
        prod_models = [m for m in models if m.status == "Production"]
        if not prod_models:
            return None
        # Sort by semantic version components
        def version_key(meta: ModelMetadata):
            return [int(x.replace("v", "")) for x in meta.version.split(".")]
        prod_models.sort(key=version_key)
        return prod_models[-1]

    def update_model_status(self, model_id: str, version: str, status: str) -> None:
        """Update model governance status (e.g. promote Experimental to Production)."""
        registry = self._read_registry()
        key = f"{model_id}:{version}"
        if key not in registry:
            raise KeyError(f"Model {key} not found in registry.")
        meta = ModelMetadata(**registry[key])
        meta.status = status
        registry[key] = meta.model_dump()
        self._write_registry(registry)
        logger.info(f"Updated status of model {key} to {status}")

    def clean_registry(self) -> None:
        """Helper to clear registry and delete stored model files (for testing/reset)."""
        self._write_registry({})
        for f in self.models_dir.glob("*.pkl"):
            try:
                f.unlink()
            except Exception:
                pass


def get_git_commit_hash() -> str:
    """Safely fetch the current git commit hash."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        return res.stdout.strip()
    except Exception:
        logger.debug("Failed to retrieve git commit hash; fallback to 'unknown'")
        return "unknown"


def save_model(
    model: Any,
    metadata_args: Dict[str, Any],
    registry: Optional[ModelRegistry] = None
) -> Path:
    """
    Serializes a model model and writes its metadata into the registry.
    Saves to <models_dir>/<model_id>_<version>.pkl.
    """
    reg = registry or ModelRegistry()
    model_id = metadata_args["model_id"]
    version = metadata_args["version"]

    # Build Pydantic model for metadata validation
    meta_args = metadata_args.copy()
    if "git_commit_hash" not in meta_args or meta_args["git_commit_hash"] == "unknown":
        meta_args["git_commit_hash"] = get_git_commit_hash()

    metadata = ModelMetadata(**meta_args)

    # Save serialized model object
    file_name = f"{model_id}_{version}.pkl"
    file_path = reg.models_dir / file_name

    # Atomically dump picklable model
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=reg.models_dir, delete=False) as tf:
            pickle.dump(model, tf)
            temp_file = Path(tf.name)
        os.replace(temp_file, file_path)
    except Exception as e:
        if temp_file and temp_file.exists():
            try:
                os.unlink(temp_file)
            except Exception:
                pass
        logger.error(f"Failed to serialize/save model file: {e}")
        raise

    # Register metadata
    reg.register_model(metadata)
    logger.info(f"Model artifact saved to: {file_path}")
    return file_path


def load_model(
    model_id: str,
    version: str,
    registry: Optional[ModelRegistry] = None,
    validation_fn: Optional[callable] = None,
    fallback_on_failure: bool = True,
) -> Tuple[Any, ModelMetadata]:
    """
    Loads a model by version. If validation_fn is provided and returns False,
    or if loading throws an exception, this function will automatically trigger
    rollback to the last known stable 'Production' model.
    """
    reg = registry or ModelRegistry()
    key = f"{model_id}:{version}"
    file_path = reg.models_dir / f"{model_id}_{version}.pkl"

    meta = reg.get_model_metadata(model_id, version)

    try:
        if not file_path.exists() or meta is None:
            raise FileNotFoundError(f"Model file or registry entry for {key} does not exist.")

        with open(file_path, "rb") as f:
            model = pickle.load(f)

        # Run optional validation hook (e.g. check for drift or performance metrics)
        if validation_fn is not None:
            if not validation_fn(model, meta):
                raise ValueError(f"Loaded model {key} failed verification checks.")

        logger.info(f"Successfully loaded model {key}")
        return model, meta

    except Exception as exc:
        logger.error(f"Error loading model {key}: {exc}")
        if fallback_on_failure:
            logger.warning("Attempting automatic rollback to last stable Production model...")
            prod_meta = reg.get_latest_production_model(model_id)
            if prod_meta is not None and prod_meta.version != version:
                logger.info(f"Rolling back to Production model version {prod_meta.version}")
                return load_model(model_id, prod_meta.version, registry=reg, validation_fn=validation_fn, fallback_on_failure=False)
            else:
                logger.error("No alternative Production model available for rollback.")
        raise

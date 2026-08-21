"""
ml_layer/pipeline.py
───────────────────
ML Pipeline Façade & High-Level Execution Engine.
Wraps feature extraction, training, evaluation, confidence, and explainability into MLPipeline.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from ml_layer.training import train, TrainResult
from ml_layer.config import MLConfig

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    target_horizon: int = 5
    cv_folds: int = 3
    model_type: str = "xgboost"
    hyperparameters: Dict[str, Any] = field(default_factory=dict)


class MLPipeline:
    """High-level machine learning pipeline façade for QuantSphereX."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.ml_config = MLConfig(base_params=self.config.hyperparameters)
        self.last_result: Optional[TrainResult] = None

    def fit_predict(self, panel_data: pd.DataFrame) -> pd.Series:
        """Fits pipeline models on panel data and generates predictions."""
        # Simple feature matrix extraction
        feature_cols = [c for c in panel_data.columns if c not in ["close", "volume", "target", "symbol"]]
        if not feature_cols:
            panel_data = panel_data.copy()
            panel_data["feature_mom"] = panel_data["close"].pct_change(5).fillna(0)
            feature_cols = ["feature_mom"]

        X = panel_data[feature_cols]
        y = panel_data["close"].pct_change(self.config.target_horizon).shift(-self.config.target_horizon).fillna(0)

        # Train model
        self.last_result = train(X, y, config=self.ml_config)
        preds = self.last_result.model.predict(X)
        return pd.Series(preds, index=X.index)

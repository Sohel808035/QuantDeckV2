"""
monitoring_layer/model_health.py
──────────────────────────────────
Institutional Model Health Monitor.

Tracks:
  - Prediction quality (rolling IC, hit rate, rank correlation)
  - Model staleness (days since last training)
  - Prediction score distribution stability
  - Feature importance drift (top-N feature rank shift)
  - Model version tracking
  - Retraining recommendations

Architecture:
  ModelHealthMonitor ──► AlertEngine
                    ──► StructuredLogger
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from monitoring_layer.config import MonitoringConfig
from monitoring_layer.alert_engine import AlertEngine, AlertSeverity

logger = logging.getLogger(__name__)


# ── Data Containers ──────────────────────────────────────────────────────────


@dataclass
class PredictionRecord:
    """Single prediction window record."""

    timestamp: float
    model_id: str
    model_version: str
    n_predictions: int
    mean_score: float
    std_score: float
    min_score: float
    max_score: float
    ic: Optional[float] = None
    hit_rate: Optional[float] = None


@dataclass
class ModelHealthState:
    """Per-model health state."""

    model_id: str
    model_version: str
    training_timestamp: Optional[float] = None
    last_prediction_timestamp: Optional[float] = None
    prediction_history: Deque[PredictionRecord] = field(default_factory=lambda: deque(maxlen=500))
    feature_importance_history: Deque[Dict[str, float]] = field(
        default_factory=lambda: deque(maxlen=30)
    )
    retrain_recommended: bool = False
    retrain_reason: str = ""


# ── Model Health Monitor ──────────────────────────────────────────────────────


class ModelHealthMonitor:
    """
    Institutional Model Health Monitor.

    Provides real-time tracking of model prediction quality, staleness,
    and feature importance stability. Integrates with AlertEngine for
    automatic breach notifications.

    Usage:
        monitor = ModelHealthMonitor()
        monitor.record_model_training("xgb_v2", "2.0.0", timestamp=time.time())
        monitor.record_predictions("xgb_v2", "2.0.0", scores)
        report = monitor.check_model_health("xgb_v2")
    """

    # Maximum staleness before issuing a WARNING (hours)
    STALENESS_WARN_HOURS: int = 24
    STALENESS_CRIT_HOURS: int = 72

    # IC thresholds
    IC_WARN_THRESHOLD: float = 0.02
    IC_CRIT_THRESHOLD: float = 0.0

    # Minimum predictions required before health scoring
    MIN_PREDICTIONS_FOR_HEALTH: int = 10

    def __init__(
        self,
        config: Optional[MonitoringConfig] = None,
        alert_engine: Optional[AlertEngine] = None,
    ):
        self.config = config or MonitoringConfig()
        self.alert_engine = alert_engine or AlertEngine(config=self.config)
        self._models: Dict[str, ModelHealthState] = {}

    # ── Registration ──────────────────────────────────────────────────────────

    def record_model_training(
        self,
        model_id: str,
        model_version: str,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Records a model training event. Call this whenever a model is retrained.

        Args:
            model_id:      Unique model identifier (e.g. 'xgb_ensemble').
            model_version: Semantic version string (e.g. '2.1.0').
            timestamp:     Unix timestamp of training (default: now).
        """
        ts = timestamp or time.time()
        state = self._get_or_create(model_id, model_version)
        state.training_timestamp = ts
        state.retrain_recommended = False
        state.retrain_reason = ""
        logger.info(
            "[ModelHealthMonitor] Model trained: %s v%s at %.0f",
            model_id, model_version, ts,
        )

    def record_predictions(
        self,
        model_id: str,
        model_version: str,
        scores: np.ndarray,
        forward_returns: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
    ) -> PredictionRecord:
        """
        Records a batch of model predictions.

        Args:
            model_id:        Model identifier.
            model_version:   Model version.
            scores:          Array of prediction scores for this window.
            forward_returns: Corresponding realized forward returns (for IC).
            timestamp:       Unix timestamp of predictions.

        Returns:
            PredictionRecord with computed metrics.
        """
        ts = timestamp or time.time()
        scores = np.array(scores, dtype=float)
        scores = scores[~np.isnan(scores)]

        if len(scores) == 0:
            logger.warning("[ModelHealthMonitor] Empty scores array — skipping record.")
            return PredictionRecord(
                timestamp=ts,
                model_id=model_id,
                model_version=model_version,
                n_predictions=0,
                mean_score=0.0,
                std_score=0.0,
                min_score=0.0,
                max_score=0.0,
            )

        ic: Optional[float] = None
        hit_rate: Optional[float] = None

        if forward_returns is not None:
            fwd = np.array(forward_returns, dtype=float)
            shared_mask = ~np.isnan(fwd) & ~np.isnan(scores[:len(fwd)])
            if shared_mask.sum() >= 5:
                corr, _ = spearmanr(scores[:len(fwd)][shared_mask], fwd[shared_mask])
                ic = float(corr) if not np.isnan(corr) else None
                # Hit rate: correct directional prediction
                correct = (np.sign(scores[:len(fwd)][shared_mask]) == np.sign(fwd[shared_mask]))
                hit_rate = float(correct.mean())

        record = PredictionRecord(
            timestamp=ts,
            model_id=model_id,
            model_version=model_version,
            n_predictions=len(scores),
            mean_score=float(np.mean(scores)),
            std_score=float(np.std(scores)),
            min_score=float(np.min(scores)),
            max_score=float(np.max(scores)),
            ic=ic,
            hit_rate=hit_rate,
        )

        state = self._get_or_create(model_id, model_version)
        state.last_prediction_timestamp = ts
        state.prediction_history.append(record)
        return record

    def record_feature_importance(
        self,
        model_id: str,
        model_version: str,
        importances: Dict[str, float],
    ) -> None:
        """
        Records a snapshot of feature importances for rank-shift monitoring.

        Args:
            model_id:     Model identifier.
            model_version: Model version.
            importances:  Dict mapping feature name → importance score.
        """
        state = self._get_or_create(model_id, model_version)
        state.feature_importance_history.append(dict(importances))

    # ── Health Checks ─────────────────────────────────────────────────────────

    def check_model_health(
        self,
        model_id: str,
        staleness_warn_hours: Optional[int] = None,
        staleness_crit_hours: Optional[int] = None,
        window: int = 20,
    ) -> Dict[str, Any]:
        """
        Full model health check for a given model.

        Args:
            model_id:              Model identifier.
            staleness_warn_hours:  Hours before WARNING on no prediction.
            staleness_crit_hours:  Hours before CRITICAL on no prediction.
            window:                Number of recent prediction records to evaluate.

        Returns:
            Health report dict with keys:
              model_id, model_version, staleness_hours, rolling_ic,
              rolling_hit_rate, score_distribution, health_score,
              retrain_recommended, issues
        """
        warn_h = staleness_warn_hours or self.STALENESS_WARN_HOURS
        crit_h = staleness_crit_hours or self.STALENESS_CRIT_HOURS

        if model_id not in self._models:
            return {
                "model_id": model_id,
                "status": "unknown",
                "health_score": 0.0,
                "issues": ["Model not registered. Call record_model_training() first."],
            }

        state = self._models[model_id]
        now = time.time()
        issues: List[str] = []
        retrain_reasons: List[str] = []

        # ── Staleness Check ───────────────────────────────────────────────────
        staleness_hours = 0.0
        if state.last_prediction_timestamp:
            staleness_hours = (now - state.last_prediction_timestamp) / 3600.0
        elif state.training_timestamp:
            staleness_hours = (now - state.training_timestamp) / 3600.0
        else:
            staleness_hours = float("inf")

        if staleness_hours >= crit_h:
            issues.append(f"CRITICAL: No predictions in {staleness_hours:.1f}h (threshold: {crit_h}h)")
            self.alert_engine.fire(
                AlertSeverity.CRITICAL, "MODEL_HEALTH", f"staleness.{model_id}",
                value=staleness_hours, threshold=crit_h,
                message=f"Model '{model_id}' stale: no predictions in {staleness_hours:.1f}h.",
            )
            retrain_reasons.append(f"stale_predictions_{staleness_hours:.0f}h")
        elif staleness_hours >= warn_h:
            issues.append(f"WARNING: No predictions in {staleness_hours:.1f}h (threshold: {warn_h}h)")
            self.alert_engine.fire(
                AlertSeverity.WARNING, "MODEL_HEALTH", f"staleness.{model_id}",
                value=staleness_hours, threshold=warn_h,
                message=f"Model '{model_id}': no recent predictions for {staleness_hours:.1f}h.",
            )

        # ── Rolling IC Check ──────────────────────────────────────────────────
        recent = list(state.prediction_history)[-window:]
        ics = [r.ic for r in recent if r.ic is not None]
        hit_rates = [r.hit_rate for r in recent if r.hit_rate is not None]

        rolling_ic: Optional[float] = None
        rolling_hit_rate: Optional[float] = None

        if ics:
            rolling_ic = float(np.mean(ics))
            if rolling_ic < self.IC_CRIT_THRESHOLD:
                issues.append(f"CRITICAL: Rolling IC = {rolling_ic:.4f} (below 0)")
                self.alert_engine.fire(
                    AlertSeverity.CRITICAL, "MODEL_HEALTH", f"ic.{model_id}",
                    value=rolling_ic, threshold=self.IC_CRIT_THRESHOLD,
                    message=f"Model '{model_id}' IC = {rolling_ic:.4f} (negative).",
                )
                retrain_reasons.append(f"negative_ic_{rolling_ic:.4f}")
            elif rolling_ic < self.IC_WARN_THRESHOLD:
                issues.append(f"WARNING: Rolling IC = {rolling_ic:.4f} (below {self.IC_WARN_THRESHOLD})")
                self.alert_engine.fire(
                    AlertSeverity.WARNING, "MODEL_HEALTH", f"ic.{model_id}",
                    value=rolling_ic, threshold=self.IC_WARN_THRESHOLD,
                    message=f"Model '{model_id}' IC = {rolling_ic:.4f} declining.",
                )

        if hit_rates:
            rolling_hit_rate = float(np.mean(hit_rates))
            if rolling_hit_rate < 0.5:
                issues.append(f"WARNING: Hit rate = {rolling_hit_rate:.2%} (below 50%)")

        # ── Score Distribution ────────────────────────────────────────────────
        score_dist: Dict[str, float] = {}
        if recent:
            mean_scores = [r.mean_score for r in recent]
            std_scores = [r.std_score for r in recent]
            score_dist = {
                "mean": round(float(np.mean(mean_scores)), 6),
                "std": round(float(np.mean(std_scores)), 6),
                "trend": round(float(np.polyfit(range(len(mean_scores)), mean_scores, 1)[0]), 8)
                if len(mean_scores) > 2
                else 0.0,
            }

        # ── Feature Importance Rank Shift ────────────────────────────────────
        fi_shift = self._compute_feature_rank_shift(state)

        # ── Health Score Computation ──────────────────────────────────────────
        health_score = self._compute_health_score(
            staleness_hours=staleness_hours,
            staleness_crit_h=crit_h,
            rolling_ic=rolling_ic,
            rolling_hit_rate=rolling_hit_rate,
            n_issues=len(issues),
        )

        # ── Retrain Recommendation ────────────────────────────────────────────
        retrain_recommended = bool(retrain_reasons) or health_score < 0.4
        if retrain_recommended and retrain_reasons:
            state.retrain_recommended = True
            state.retrain_reason = "; ".join(retrain_reasons)

        return {
            "model_id": model_id,
            "model_version": state.model_version,
            "staleness_hours": round(staleness_hours, 2) if staleness_hours != float("inf") else None,
            "rolling_ic": round(rolling_ic, 4) if rolling_ic is not None else None,
            "rolling_hit_rate": round(rolling_hit_rate, 4) if rolling_hit_rate is not None else None,
            "score_distribution": score_dist,
            "feature_importance_rank_shift": fi_shift,
            "n_predictions_window": len(recent),
            "health_score": round(health_score, 3),
            "retrain_recommended": retrain_recommended,
            "retrain_reason": state.retrain_reason,
            "issues": issues,
        }

    def check_all_models(self, window: int = 20) -> Dict[str, Any]:
        """
        Runs health checks on all registered models.

        Returns:
            Dict mapping model_id → health report.
        """
        reports = {}
        for model_id in self._models:
            reports[model_id] = self.check_model_health(model_id, window=window)
        return reports

    # ── Internal Helpers ──────────────────────────────────────────────────────

    def _get_or_create(self, model_id: str, model_version: str) -> ModelHealthState:
        if model_id not in self._models:
            self._models[model_id] = ModelHealthState(
                model_id=model_id,
                model_version=model_version,
            )
        return self._models[model_id]

    @staticmethod
    def _compute_feature_rank_shift(state: ModelHealthState) -> Optional[float]:
        """
        Computes the Spearman rank correlation between the first and last
        feature importance snapshots. Low correlation indicates feature importance drift.

        Returns:
            Rank correlation (1.0 = no shift, 0.0 = complete shuffle) or None.
        """
        history = list(state.feature_importance_history)
        if len(history) < 2:
            return None

        first = history[0]
        last = history[-1]
        common_features = [f for f in first if f in last]

        if len(common_features) < 3:
            return None

        first_vals = [first[f] for f in common_features]
        last_vals = [last[f] for f in common_features]
        corr, _ = spearmanr(first_vals, last_vals)
        return round(float(corr), 4) if not np.isnan(corr) else None

    @staticmethod
    def _compute_health_score(
        staleness_hours: float,
        staleness_crit_h: float,
        rolling_ic: Optional[float],
        rolling_hit_rate: Optional[float],
        n_issues: int,
    ) -> float:
        """
        Computes a composite health score in [0, 1].
          1.0 = perfect health
          0.0 = critical failure
        """
        score = 1.0

        # Staleness penalty
        if staleness_hours != float("inf"):
            staleness_ratio = min(staleness_hours / max(staleness_crit_h, 1), 1.0)
            score -= 0.30 * staleness_ratio
        else:
            score -= 0.50

        # IC quality penalty
        if rolling_ic is not None:
            if rolling_ic < 0:
                score -= 0.35
            elif rolling_ic < 0.02:
                score -= 0.20
            elif rolling_ic < 0.05:
                score -= 0.10

        # Hit rate penalty
        if rolling_hit_rate is not None:
            if rolling_hit_rate < 0.45:
                score -= 0.20
            elif rolling_hit_rate < 0.50:
                score -= 0.10

        # Issue count penalty (minor)
        score -= 0.05 * n_issues

        return max(0.0, min(1.0, score))

    # ── State Inspection ──────────────────────────────────────────────────────

    def registered_models(self) -> List[str]:
        """Returns list of all registered model IDs."""
        return list(self._models.keys())

    def reset(self, model_id: Optional[str] = None) -> None:
        """Clears history for a specific model or all models."""
        if model_id:
            self._models.pop(model_id, None)
        else:
            self._models.clear()

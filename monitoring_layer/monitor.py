"""
monitoring_layer/monitor.py
─────────────────────────────
Master MonitoringLayer Orchestrator — Upgraded Institutional Version.

Single entry point wiring ALL monitoring components:
  - DataQualityMonitor     : Data schema, missing, staleness, outliers
  - DriftMonitor           : Feature and prediction distributional shift
  - SystemHealthMonitor    : CPU, memory, latency, errors
  - StrategyMonitor        : Rolling Sharpe, Rolling IC, Drawdown
  - ModelHealthMonitor     : Model quality, staleness, retrain recommendations
  - PortfolioRiskMonitor   : VaR, CVaR, concentration, leverage, turnover
  - MarketRegimeMonitor    : Volatility, trend, correlation regimes
  - DataFreshnessMonitor   : Feed staleness and data gap detection
  - AlertEngine            : Alert generation, deduplication, dispatch
  - StructuredLogger       : JSON-formatted rotating log
  - MonitoringDashboard    : Rich terminal dashboard
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from monitoring_layer.config import MonitoringConfig
from monitoring_layer.alert_engine import AlertEngine, Alert
from monitoring_layer.data_quality import DataQualityMonitor
from monitoring_layer.drift import DriftMonitor
from monitoring_layer.system_health import SystemHealthMonitor
from monitoring_layer.strategy_monitor import StrategyMonitor
from monitoring_layer.model_health import ModelHealthMonitor
from monitoring_layer.portfolio_risk_monitor import PortfolioRiskMonitor
from monitoring_layer.market_regime_monitor import MarketRegimeMonitor
from monitoring_layer.data_freshness_monitor import DataFreshnessMonitor, FeedFrequency
from monitoring_layer.logger import StructuredLogger
from monitoring_layer.dashboard import MonitoringDashboard

logger = logging.getLogger(__name__)


class MonitoringLayer:
    """
    QuantSphereX Institutional Monitoring Layer — unified system diagnostics.

    Components:
      - DataQualityMonitor     : Data schema, missing, staleness, outliers
      - DriftMonitor           : Feature and prediction distributional shift
      - SystemHealthMonitor    : CPU, memory, latency, errors
      - StrategyMonitor        : Rolling Sharpe, Rolling IC, Drawdown
      - ModelHealthMonitor     : Model quality, staleness, retrain recommendations
      - PortfolioRiskMonitor   : VaR, CVaR, concentration, leverage, turnover
      - MarketRegimeMonitor    : Volatility, trend, correlation regimes
      - DataFreshnessMonitor   : Feed staleness and data gap detection
      - AlertEngine            : Alert generation, deduplication, dispatch
      - StructuredLogger       : JSON-formatted rotating log
      - MonitoringDashboard    : Rich terminal dashboard

    Usage:
        ml = MonitoringLayer()
        ml.check_data_quality(df, "prices")
        ml.check_system_health()
        ml.check_model_health("xgb_v2")
        ml.check_portfolio_risk(weights, returns_df)
        ml.render_dashboard()
    """

    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()
        self.alert_engine = AlertEngine(config=self.config)

        # Core monitors
        self.data_quality = DataQualityMonitor(self.config, self.alert_engine)
        self.drift = DriftMonitor(self.config, self.alert_engine)
        self.system = SystemHealthMonitor(self.config, self.alert_engine)
        self.strategy = StrategyMonitor(self.config, self.alert_engine)

        # New institutional monitors
        self.model_health = ModelHealthMonitor(self.config, self.alert_engine)
        self.portfolio_risk = PortfolioRiskMonitor(self.config, self.alert_engine)
        self.market_regime = MarketRegimeMonitor(self.config, self.alert_engine)
        self.data_freshness = DataFreshnessMonitor(self.config, self.alert_engine)

        # Logging and dashboard
        self.slog = StructuredLogger(
            name=f"{self.config.service_name}.monitoring",
            config=self.config,
        )
        self.dashboard = MonitoringDashboard(
            service_name=f"{self.config.service_name} Monitoring Layer"
        )

        # Last-known state cache for dashboard
        self._last_health: Dict[str, Any] = {}
        self._last_dq: Dict[str, Any] = {}
        self._last_drift: Dict[str, Any] = {}
        self._last_strategy: Dict[str, Any] = {}
        self._last_model_health: Dict[str, Any] = {}
        self._last_portfolio_risk: Dict[str, Any] = {}
        self._last_market_regime: Dict[str, Any] = {}
        self._last_freshness: Dict[str, Any] = {}

    # ── Data Quality ─────────────────────────────────────────────────────────

    def check_data_quality(self, df: pd.DataFrame, feed_name: str = "feed") -> Dict[str, Any]:
        """Runs all data quality checks on the given DataFrame."""
        with self.system.track_latency(f"data_quality.{feed_name}"):
            report = self.data_quality.check(df, feed_name)
        self._last_dq = report
        self.slog.info("data_quality_check", feed=feed_name, passed=report.get("passed"), failed=report.get("failed_checks"))
        return report

    # ── Feature & Prediction Drift ────────────────────────────────────────────

    def check_feature_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Checks for feature distributional drift between reference and current windows."""
        with self.system.track_latency("drift.feature"):
            report = self.drift.check_feature_drift(reference_df, current_df, feature_cols)
        self._last_drift = report
        self.slog.info("feature_drift_check", drifted_features=report.get("drifted_features"), drift_rate=report.get("drift_rate"))
        return report

    def check_prediction_drift(
        self,
        reference_predictions: np.ndarray,
        current_predictions: np.ndarray,
        model_name: str = "model",
    ) -> Dict[str, Any]:
        """Checks for prediction score distributional drift."""
        with self.system.track_latency(f"drift.prediction.{model_name}"):
            report = self.drift.check_prediction_drift(reference_predictions, current_predictions, model_name)
        self.slog.info("prediction_drift_check", model=model_name, psi=report.get("psi"), drifted=report.get("drifted"))
        return report

    # ── System Health ─────────────────────────────────────────────────────────

    def check_system_health(self) -> Dict[str, Any]:
        """Checks current CPU, memory, and returns system health report."""
        report = self.system.check_cpu_memory()
        report["latency_summary"] = self.system.latency_summary()
        report["error_summary"] = self.system.error_summary()
        self._last_health = report
        self.slog.info("system_health_check", cpu_pct=report.get("cpu_pct"), memory_pct=report.get("memory_pct"))
        return report

    # ── Strategy Monitor ──────────────────────────────────────────────────────

    def check_rolling_sharpe(
        self, daily_returns: pd.Series, window: Optional[int] = None
    ) -> Dict[str, Any]:
        """Checks rolling Sharpe ratio and fires alert if below threshold."""
        report = self.strategy.check_rolling_sharpe(daily_returns, window)
        self._last_strategy.update({k: v for k, v in report.items() if not isinstance(v, pd.Series)})
        self.slog.info("rolling_sharpe_check", latest=report.get("latest_sharpe"), breach=report.get("breach"))
        return report

    def check_rolling_ic(
        self,
        alpha_scores: pd.DataFrame,
        forward_returns: pd.DataFrame,
        window: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Checks rolling IC and fires alert if below threshold."""
        report = self.strategy.check_rolling_ic(alpha_scores, forward_returns, window)
        self._last_strategy.update({k: v for k, v in report.items() if not isinstance(v, pd.Series)})
        self.slog.info("rolling_ic_check", latest=report.get("latest_ic"), breach=report.get("breach"))
        return report

    def check_drawdown(self, equity_curve: pd.Series) -> Dict[str, Any]:
        """Checks current drawdown and fires alert if breach threshold exceeded."""
        report = self.strategy.check_drawdown(equity_curve)
        self._last_strategy.update({k: v for k, v in report.items() if not isinstance(v, pd.Series)})
        self.slog.info("drawdown_check", current=report.get("current_drawdown"), breach=report.get("breach"))
        return report

    # ── Model Health ──────────────────────────────────────────────────────────

    def record_model_training(
        self,
        model_id: str,
        model_version: str,
        timestamp: Optional[float] = None,
    ) -> None:
        """Records a model training event for staleness tracking."""
        self.model_health.record_model_training(model_id, model_version, timestamp)
        self.slog.info("model_training_recorded", model_id=model_id, version=model_version)

    def record_predictions(
        self,
        model_id: str,
        model_version: str,
        scores: np.ndarray,
        forward_returns: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Records model predictions for quality tracking."""
        record = self.model_health.record_predictions(
            model_id, model_version, scores, forward_returns, timestamp
        )
        self.slog.info(
            "predictions_recorded",
            model_id=model_id,
            n=record.n_predictions,
            ic=record.ic,
        )
        return {
            "model_id": record.model_id,
            "n_predictions": record.n_predictions,
            "mean_score": record.mean_score,
            "ic": record.ic,
        }

    def check_model_health(
        self,
        model_id: str,
        window: int = 20,
    ) -> Dict[str, Any]:
        """Full health check for a specific model."""
        with self.system.track_latency(f"model_health.{model_id}"):
            report = self.model_health.check_model_health(model_id, window=window)
        self._last_model_health[model_id] = report
        self.slog.info(
            "model_health_check",
            model_id=model_id,
            health_score=report.get("health_score"),
            retrain=report.get("retrain_recommended"),
        )
        return report

    def check_all_model_health(self, window: int = 20) -> Dict[str, Any]:
        """Runs health checks on all registered models."""
        reports = self.model_health.check_all_models(window=window)
        self._last_model_health = reports
        self.slog.info("all_model_health_check", n_models=len(reports))
        return reports

    # ── Portfolio Risk ────────────────────────────────────────────────────────

    def check_portfolio_risk(
        self,
        weights: pd.Series,
        returns_df: pd.DataFrame,
        prev_weights: Optional[pd.Series] = None,
        sector_map: Optional[Dict[str, str]] = None,
        lookback_days: int = 252,
    ) -> Dict[str, Any]:
        """Full portfolio risk check: VaR, concentration, leverage, turnover."""
        with self.system.track_latency("portfolio_risk"):
            report = self.portfolio_risk.check(
                weights, returns_df, prev_weights, sector_map, lookback_days
            )
        self._last_portfolio_risk = report
        self.slog.info(
            "portfolio_risk_check",
            var=report.get("historical_var"),
            hhi=report.get("hhi"),
            gross_exposure=report.get("gross_exposure"),
        )
        return report

    # ── Market Regime ─────────────────────────────────────────────────────────

    def detect_market_regime(
        self,
        market_returns: pd.Series,
        multi_asset_returns: Optional[pd.DataFrame] = None,
        vol_window: int = 21,
        price_series: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """Detects current market regime (volatility, trend, correlation)."""
        with self.system.track_latency("market_regime"):
            report = self.market_regime.detect_regime(
                market_returns, multi_asset_returns, vol_window, price_series
            )
        self._last_market_regime = report
        self.slog.info(
            "market_regime_detection",
            vol_regime=report.get("volatility_regime"),
            trend_regime=report.get("trend_regime"),
            composite=report.get("regime_composite"),
        )
        return report

    # ── Data Freshness ────────────────────────────────────────────────────────

    def record_feed_update(
        self,
        feed_name: str,
        data: Optional[pd.DataFrame] = None,
        n_rows: Optional[int] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """Records a data feed update for freshness tracking."""
        self.data_freshness.record_update(feed_name, data, n_rows, timestamp)

    def check_data_freshness(self) -> Dict[str, Any]:
        """Checks freshness of all registered data feeds."""
        report = self.data_freshness.check_all_feeds()
        self._last_freshness = report
        self.slog.info(
            "data_freshness_check",
            fresh=report.get("summary", {}).get("fresh"),
            stale=report.get("summary", {}).get("stale"),
            critical=report.get("summary", {}).get("critical"),
        )
        return report

    def register_feed(
        self,
        name: str,
        frequency: FeedFrequency = FeedFrequency.DAILY,
        warn_after_seconds: int = 86400,
        crit_after_seconds: int = 172800,
    ) -> None:
        """Registers a custom data feed for freshness monitoring."""
        self.data_freshness.register_feed(
            name, frequency, warn_after_seconds, crit_after_seconds
        )

    # ── Full Health Check ─────────────────────────────────────────────────────

    def full_health_check(
        self,
        data_df: Optional[pd.DataFrame] = None,
        daily_returns: Optional[pd.Series] = None,
        equity_curve: Optional[pd.Series] = None,
        market_returns: Optional[pd.Series] = None,
        portfolio_weights: Optional[pd.Series] = None,
        returns_df: Optional[pd.DataFrame] = None,
        reference_df: Optional[pd.DataFrame] = None,
        current_df: Optional[pd.DataFrame] = None,
        reference_predictions: Optional[np.ndarray] = None,
        current_predictions: Optional[np.ndarray] = None,
        alpha_scores: Optional[pd.DataFrame] = None,
        forward_returns: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Runs all monitoring checks in a single call.
        Returns a consolidated health report.
        """
        results: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "service": self.config.service_name,
        }

        results["system_health"] = self.check_system_health()

        if data_df is not None:
            results["data_quality"] = self.check_data_quality(data_df)

        if daily_returns is not None:
            results["rolling_sharpe"] = self.check_rolling_sharpe(daily_returns)

        if alpha_scores is not None and forward_returns is not None:
            results["rolling_ic"] = self.check_rolling_ic(alpha_scores, forward_returns)

        if equity_curve is not None:
            results["drawdown"] = self.check_drawdown(equity_curve)

        if market_returns is not None:
            results["market_regime"] = self.detect_market_regime(market_returns)

        if portfolio_weights is not None and returns_df is not None:
            results["portfolio_risk"] = self.check_portfolio_risk(portfolio_weights, returns_df)

        if reference_df is not None and current_df is not None:
            results["feature_drift"] = self.check_feature_drift(reference_df, current_df)

        if reference_predictions is not None and current_predictions is not None:
            results["prediction_drift"] = self.check_prediction_drift(reference_predictions, current_predictions)

        # Data freshness
        results["data_freshness"] = self.check_data_freshness()

        # Model health (all registered models)
        if self.model_health.registered_models():
            results["model_health"] = self.check_all_model_health()

        alert_summary = self.alert_engine.summary()
        results["alerts_summary"] = alert_summary

        # Overall health
        critical_count = alert_summary.get("CRITICAL", 0)
        warning_count = alert_summary.get("WARNING", 0)
        if critical_count > 0:
            results["overall_health"] = "CRITICAL"
        elif warning_count > 0:
            results["overall_health"] = "WARNING"
        else:
            results["overall_health"] = "HEALTHY"

        self.slog.info("full_health_check", overall=results["overall_health"])
        return results

    # ── Dashboard ─────────────────────────────────────────────────────────────

    def render_dashboard(self) -> None:
        """Renders the monitoring dashboard using last-known state of all monitors."""
        self.dashboard.render(
            health_report=self._last_health,
            data_quality_report=self._last_dq,
            drift_report=self._last_drift,
            strategy_report=self._last_strategy,
            recent_alerts=self.alert_engine.recent_alerts(n=10),
            model_health_report=self._last_model_health,
            portfolio_risk_report=self._last_portfolio_risk,
            market_regime_report=self._last_market_regime,
            freshness_report=self._last_freshness,
        )

    # ── Alert Helpers ─────────────────────────────────────────────────────────

    def recent_alerts(self, n: int = 20) -> List[Alert]:
        return self.alert_engine.recent_alerts(n)

    def alert_summary(self) -> Dict[str, int]:
        return self.alert_engine.summary()

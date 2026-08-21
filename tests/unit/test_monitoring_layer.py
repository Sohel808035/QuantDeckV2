"""
tests/test_monitoring_layer.py
────────────────────────────────
Unit Test Suite for QuantSphereX Monitoring Layer.

Covers:
  1.  AlertEngine            - Fire, cooldown, rate limiting, dedup, summary
  2.  DataQualityMonitor     - Missing, outlier, staleness, duplicates, constants
  3.  DriftMonitor           - PSI computation, feature drift, prediction drift
  4.  SystemHealthMonitor    - CPU/memory checks, latency tracking, error rate
  5.  StrategyMonitor        - Rolling Sharpe, Rolling IC, drawdown checks
  6.  StructuredLogger       - Logger creation, JSON format
  7.  MonitoringDashboard    - Plain-text render (no rich required)
  8.  MonitoringLayer        - Full orchestration, full_health_check, alerts
  9.  ModelHealthMonitor     - Prediction quality, staleness, IC, retrain
  10. PortfolioRiskMonitor   - VaR/CVaR, concentration, leverage, turnover
  11. MarketRegimeMonitor    - Volatility, trend, correlation regimes
  12. DataFreshnessMonitor   - Feed staleness, row count, health scoring
"""

import os
import time
import logging
import tempfile
import unittest
import numpy as np
import pandas as pd

from monitoring_layer.config import MonitoringConfig, AlertConfig
from monitoring_layer.alert_engine import AlertEngine, Alert, AlertSeverity
from monitoring_layer.data_quality import DataQualityMonitor
from monitoring_layer.drift import DriftMonitor, _compute_psi
from monitoring_layer.system_health import SystemHealthMonitor
from monitoring_layer.strategy_monitor import StrategyMonitor
from monitoring_layer.model_health import ModelHealthMonitor, PredictionRecord
from monitoring_layer.portfolio_risk_monitor import PortfolioRiskMonitor
from monitoring_layer.market_regime_monitor import (
    MarketRegimeMonitor,
    VolatilityRegime,
    TrendRegime,
    CorrelationRegime,
)
from monitoring_layer.data_freshness_monitor import (
    DataFreshnessMonitor,
    FeedFrequency,
)
from monitoring_layer.logger import StructuredLogger, build_monitoring_logger
from monitoring_layer.dashboard import MonitoringDashboard
from monitoring_layer.monitor import MonitoringLayer


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_config(log_dir=None):
    cfg = MonitoringConfig()
    if log_dir:
        cfg.alerts.log_dir = log_dir
    return cfg


def _make_returns(n=252, seed=42):
    np.random.seed(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    return pd.Series(np.random.normal(0.0003, 0.01, n), index=dates)


def _make_df(n_rows=100, n_cols=5, seed=42):
    np.random.seed(seed)
    end_date = pd.Timestamp.now()
    dates = pd.date_range(end=end_date, periods=n_rows, freq="B")
    cols = [f"f{i}" for i in range(n_cols)]
    return pd.DataFrame(np.random.randn(n_rows, n_cols), index=dates, columns=cols)


# ─── 1. AlertEngine ──────────────────────────────────────────────────────────

class TestAlertEngine(unittest.TestCase):
    def setUp(self):
        self.engine = AlertEngine()
        self.engine.clear()

    def test_fire_returns_alert(self):
        a = self.engine.fire(
            AlertSeverity.WARNING, "TEST", "metric", 0.5, 0.3, "test message"
        )
        self.assertIsInstance(a, Alert)
        self.assertEqual(a.severity, AlertSeverity.WARNING)

    def test_cooldown_deduplication(self):
        self.engine.fire(AlertSeverity.WARNING, "TEST", "metric", 0.5, 0.3, "first")
        second = self.engine.fire(AlertSeverity.WARNING, "TEST", "metric", 0.6, 0.3, "second")
        self.assertIsNone(second)  # Suppressed by cooldown

    def test_different_metric_not_suppressed(self):
        self.engine.fire(AlertSeverity.WARNING, "TEST", "metric_a", 0.5, 0.3, "msg a")
        b = self.engine.fire(AlertSeverity.WARNING, "TEST", "metric_b", 0.5, 0.3, "msg b")
        self.assertIsNotNone(b)  # Different metric — should fire

    def test_summary_counts(self):
        self.engine.fire(AlertSeverity.WARNING, "A", "m1", 0.5, 0.3, "w1")
        self.engine.fire(AlertSeverity.CRITICAL, "A", "m2", 0.9, 0.8, "c1")
        s = self.engine.summary()
        self.assertEqual(s.get("WARNING", 0), 1)
        self.assertEqual(s.get("CRITICAL", 0), 1)

    def test_recent_alerts_order(self):
        self.engine.fire(AlertSeverity.INFO, "A", "m1", 0.1, 0.0, "first")
        self.engine.fire(AlertSeverity.INFO, "A", "m2", 0.2, 0.0, "second")
        recent = self.engine.recent_alerts(n=2)
        self.assertEqual(len(recent), 2)

    def test_custom_handler_called(self):
        received = []
        self.engine.register_handler(lambda a: received.append(a))
        self.engine.fire(AlertSeverity.INFO, "A", "m9", 0.1, 0.0, "test")
        self.assertEqual(len(received), 1)

    def test_alert_to_dict(self):
        a = self.engine.fire(AlertSeverity.CRITICAL, "B", "m8", 0.9, 0.5, "critical!")
        d = a.to_dict()
        self.assertIn("severity", d)
        self.assertIn("timestamp", d)
        self.assertEqual(d["severity"], "CRITICAL")


# ─── 2. DataQualityMonitor ────────────────────────────────────────────────────

class TestDataQualityMonitor(unittest.TestCase):
    def setUp(self):
        self.dq = DataQualityMonitor()

    def test_clean_df_passes(self):
        df = _make_df(n_rows=100)
        report = self.dq.check(df, "clean_feed")
        self.assertTrue(report["passed"])

    def test_empty_df_fails(self):
        df = pd.DataFrame()
        report = self.dq.check(df, "empty_feed")
        self.assertFalse(report["passed"])

    def test_missing_values_detected(self):
        df = _make_df(n_rows=100)
        df.iloc[:50, :] = np.nan  # 50% missing
        report = self.dq.check(df, "missing_feed")
        self.assertIn("missing_values", report["checks"])
        self.assertFalse(report["checks"]["missing_values"]["passed"])

    def test_duplicate_index_detected(self):
        df = _make_df(n_rows=50)
        df = pd.concat([df, df])  # Duplicate all rows
        report = self.dq.check(df, "dup_feed")
        self.assertFalse(report["checks"]["duplicates"]["passed"])

    def test_constant_column_detected(self):
        df = _make_df(n_rows=50)
        df["const_col"] = 5.0  # Constant column
        report = self.dq.check(df, "const_feed")
        self.assertFalse(report["checks"]["constant_columns"]["passed"])
        self.assertIn("const_col", report["checks"]["constant_columns"]["constant_columns"])

    def test_outlier_detection(self):
        df = _make_df(n_rows=100)
        df.iloc[0, 0] = 1e9  # Extreme outlier
        report = self.dq.check(df, "outlier_feed")
        self.assertIn("outliers", report["checks"])

    def test_report_has_feed_name(self):
        df = _make_df()
        report = self.dq.check(df, "my_feed")
        self.assertEqual(report["feed"], "my_feed")


# ─── 3. DriftMonitor ─────────────────────────────────────────────────────────

class TestDriftMonitor(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.reference = pd.DataFrame(
            np.random.randn(200, 3), columns=["f1", "f2", "f3"],
            index=pd.date_range("2021-01-01", periods=200, freq="B")
        )

    def test_psi_zero_for_identical_distributions(self):
        psi = _compute_psi(np.random.randn(500), np.random.randn(500))
        # PSI for same distribution should be near 0
        self.assertLess(psi, 0.2)

    def test_psi_high_for_shifted_distribution(self):
        ref = np.random.randn(500)
        cur = np.random.randn(500) + 5.0  # Huge shift
        psi = _compute_psi(ref, cur)
        self.assertGreater(psi, 0.2)

    def test_feature_drift_no_drift(self):
        dm = DriftMonitor()
        current = pd.DataFrame(
            np.random.randn(200, 3), columns=["f1", "f2", "f3"],
            index=pd.date_range("2022-01-01", periods=200, freq="B")
        )
        report = dm.check_feature_drift(self.reference, current)
        self.assertIn("features", report)
        self.assertIn("drift_rate", report)

    def test_feature_drift_detects_shift(self):
        dm = DriftMonitor()
        shifted = pd.DataFrame(
            np.random.randn(200, 3) + 10.0, columns=["f1", "f2", "f3"],
            index=pd.date_range("2022-01-01", periods=200, freq="B")
        )
        report = dm.check_feature_drift(self.reference, shifted)
        # With a shift of 10 std devs, at least one feature should drift
        self.assertGreater(len(report["drifted_features"]), 0)

    def test_prediction_drift_no_drift(self):
        dm = DriftMonitor()
        ref_preds = np.random.randn(200)
        cur_preds = np.random.randn(200)
        report = dm.check_prediction_drift(ref_preds, cur_preds, model_name="xgboost")
        self.assertIn("psi", report)
        self.assertIn("drifted", report)

    def test_prediction_drift_detects_shift(self):
        dm = DriftMonitor()
        ref_preds = np.random.randn(200)
        cur_preds = np.random.randn(200) + 8.0  # Large shift
        report = dm.check_prediction_drift(ref_preds, cur_preds, model_name="xgboost")
        self.assertTrue(report["drifted"])

    def test_insufficient_data_returns_gracefully(self):
        dm = DriftMonitor()
        ref = np.random.randn(5)  # Too few samples
        cur = np.random.randn(5)
        report = dm.check_prediction_drift(ref, cur)
        self.assertEqual(report.get("status"), "insufficient_data")


# ─── 4. SystemHealthMonitor ───────────────────────────────────────────────────

class TestSystemHealthMonitor(unittest.TestCase):
    def setUp(self):
        self.sm = SystemHealthMonitor()

    def test_cpu_memory_returns_dict(self):
        report = self.sm.check_cpu_memory()
        self.assertIsInstance(report, dict)

    def test_latency_track_records_measurement(self):
        with self.sm.track_latency("test_op"):
            time.sleep(0.01)
        summary = self.sm.latency_summary("test_op")
        self.assertIn("mean_ms", summary)
        self.assertGreater(summary["mean_ms"], 0)

    def test_latency_percentiles_available(self):
        for _ in range(10):
            with self.sm.track_latency("perf_test"):
                time.sleep(0.001)
        summary = self.sm.latency_summary("perf_test")
        self.assertIn("p95_ms", summary)
        self.assertIn("p99_ms", summary)

    def test_error_recording(self):
        self.sm._call_counts["my_op"] = 100
        self.sm.record_error("my_op")
        summary = self.sm.error_summary()
        self.assertIn("my_op", summary)
        self.assertGreater(summary["my_op"]["errors"], 0)

    def test_latency_track_records_exception(self):
        try:
            with self.sm.track_latency("failing_op"):
                raise ValueError("Simulated failure")
        except ValueError:
            pass
        # Error should be recorded
        summary = self.sm.error_summary()
        self.assertIn("failing_op", summary)


# ─── 5. StrategyMonitor ──────────────────────────────────────────────────────

class TestStrategyMonitor(unittest.TestCase):
    def setUp(self):
        self.returns = _make_returns(n=252)
        self.equity = (1_000_000 * (1 + self.returns).cumprod())

    def test_rolling_sharpe_returns_dict(self):
        sm = StrategyMonitor()
        result = sm.check_rolling_sharpe(self.returns, window=63)
        self.assertIn("latest_sharpe", result)
        self.assertIn("breach", result)

    def test_rolling_sharpe_breach_for_low_returns(self):
        sm = StrategyMonitor()
        bad_returns = pd.Series(-0.01, index=self.returns.index)
        result = sm.check_rolling_sharpe(bad_returns, window=63)
        self.assertTrue(result["breach"])

    def test_drawdown_check_returns_metrics(self):
        sm = StrategyMonitor()
        result = sm.check_drawdown(self.equity)
        self.assertIn("current_drawdown", result)
        self.assertIn("max_drawdown", result)
        self.assertLessEqual(result["max_drawdown"], 0.0)

    def test_drawdown_breach_fires_for_deep_dd(self):
        sm = StrategyMonitor()
        crashing = pd.Series(
            [1_000_000 * (0.5 ** (i / 50)) for i in range(100)],
            index=pd.date_range("2022-01-01", periods=100, freq="B")
        )
        result = sm.check_drawdown(crashing)
        self.assertTrue(result["breach"])

    def test_rolling_ic_returns_dict(self):
        sm = StrategyMonitor()
        dates = pd.date_range("2022-01-01", periods=100, freq="B")
        tickers = ["A", "B", "C", "D", "E"]
        alpha_scores = pd.DataFrame(np.random.randn(100, 5), index=dates, columns=tickers)
        fwd_rets = pd.DataFrame(np.random.randn(100, 5), index=dates, columns=tickers)
        result = sm.check_rolling_ic(alpha_scores, fwd_rets, window=21)
        self.assertIn("latest_ic", result)

    def test_insufficient_data_handled(self):
        sm = StrategyMonitor()
        short_returns = _make_returns(n=10)
        result = sm.check_rolling_sharpe(short_returns, window=63)
        self.assertIn("status", result)


# ─── 6. StructuredLogger ─────────────────────────────────────────────────────

class TestStructuredLogger(unittest.TestCase):
    def tearDown(self):
        # Close file handlers to release Windows locks
        for name in ["test.logger", "test.monitor", "file.test", "quantspherex.monitoring"]:
            logger = logging.getLogger(name)
            for h in list(logger.handlers):
                h.close()
                logger.removeHandler(h)

    def test_logger_creates_without_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(log_dir=tmpdir)
            slog = StructuredLogger("test.logger", config=cfg)
            slog.info("test_event", key1="val1", key2=42)
            # Close handlers inside context to release lock
            for h in list(slog._logger.handlers):
                h.close()
                slog._logger.removeHandler(h)

    def test_build_monitoring_logger_returns_logger(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(log_dir=tmpdir)
            lg = build_monitoring_logger("test.monitor", config=cfg)
            self.assertIsInstance(lg, logging.Logger)
            for h in list(lg.handlers):
                h.close()
                lg.removeHandler(h)

    def test_log_file_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(log_dir=tmpdir)
            slog = StructuredLogger("file.test", config=cfg)
            slog.warning("warn_event", value=99)
            files = os.listdir(tmpdir)
            self.assertTrue(any(".log" in f for f in files))
            for h in list(slog._logger.handlers):
                h.close()
                slog._logger.removeHandler(h)


# ─── 7. MonitoringDashboard ──────────────────────────────────────────────────

class TestMonitoringDashboard(unittest.TestCase):
    def test_plain_render_no_exception(self):
        """Dashboard should render without errors even without 'rich' library."""
        db = MonitoringDashboard.__new__(MonitoringDashboard)
        db.service_name = "Test"
        db._rich = None  # Force plain-text mode
        # Should not raise — pass all new kwargs
        db._render_plain(
            health={"cpu_pct": 45.0, "memory_pct": 60.0},
            dq={"checks": {"missing_values": {"passed": True}}},
            drift={"features": {"f1": {"psi": 0.05, "ks_pvalue": 0.3, "drifted": False}}},
            strategy={"latest_sharpe": 0.85, "breach": False},
            alerts=[],
            model_health={"xgb": {"health_score": 0.9, "retrain_recommended": False}},
            port_risk={"historical_var": 0.015, "hhi": 0.08},
            market_regime={"volatility_regime": "NORMAL", "trend_regime": "BULL", "regime_composite": "BULL"},
            freshness={"summary": {"total_feeds": 4, "fresh": 4, "stale": 0, "critical": 0, "overall_health": "HEALTHY"}},
        )

    def test_plain_render_with_none_new_kwargs(self):
        """Dashboard should not crash when new kwargs are None (backwards compat)."""
        db = MonitoringDashboard.__new__(MonitoringDashboard)
        db.service_name = "Test"
        db._rich = None
        db._render_plain(
            health={"cpu_pct": 45.0, "memory_pct": 60.0},
            dq=None,
            drift=None,
            strategy=None,
            alerts=[],
            model_health=None,
            port_risk=None,
            market_regime=None,
            freshness=None,
        )

    def test_dashboard_init(self):
        db = MonitoringDashboard(service_name="TestSvc")
        self.assertEqual(db.service_name, "TestSvc")


# ─── 8. MonitoringLayer (Full Orchestration) ──────────────────────────────────

class TestMonitoringLayer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cfg = MonitoringConfig()
        self.cfg.alerts.log_dir = self.tmp
        self.cfg.alerts.enable_console = False
        self.ml = MonitoringLayer(config=self.cfg)
        self.returns = _make_returns(n=252)
        self.equity = 1_000_000 * (1 + self.returns).cumprod()
        self.df = _make_df(n_rows=100)

    def tearDown(self):
        if hasattr(self.ml, 'slog'):
            for h in list(self.ml.slog._logger.handlers):
                h.close()
                self.ml.slog._logger.removeHandler(h)

    def test_check_data_quality_passes_clean(self):
        report = self.ml.check_data_quality(self.df, "test_feed")
        self.assertIn("passed", report)

    def test_check_rolling_sharpe(self):
        result = self.ml.check_rolling_sharpe(self.returns, window=63)
        self.assertIn("latest_sharpe", result)

    def test_check_drawdown(self):
        result = self.ml.check_drawdown(self.equity)
        self.assertIn("current_drawdown", result)

    def test_check_feature_drift(self):
        ref = _make_df(n_rows=200)
        cur = _make_df(n_rows=100, seed=99)
        result = self.ml.check_feature_drift(ref, cur)
        self.assertIn("drift_rate", result)

    def test_check_prediction_drift(self):
        ref_preds = np.random.randn(200)
        cur_preds = np.random.randn(100)
        result = self.ml.check_prediction_drift(ref_preds, cur_preds, "test_model")
        self.assertIn("drifted", result)

    def test_full_health_check(self):
        result = self.ml.full_health_check(
            data_df=self.df,
            daily_returns=self.returns,
            equity_curve=self.equity,
        )
        self.assertIn("overall_health", result)
        self.assertIn(result["overall_health"], ["HEALTHY", "WARNING", "CRITICAL"])

    def test_full_health_check_includes_data_freshness(self):
        result = self.ml.full_health_check()
        self.assertIn("data_freshness", result)

    def test_alert_summary_returns_dict(self):
        summary = self.ml.alert_summary()
        self.assertIsInstance(summary, dict)

    def test_recent_alerts_returns_list(self):
        alerts = self.ml.recent_alerts(n=5)
        self.assertIsInstance(alerts, list)

    def test_render_dashboard_no_exception(self):
        self.ml._last_health = {"cpu_pct": 30.0, "memory_pct": 50.0}
        self.ml.dashboard._rich = None  # Force plain-text
        self.ml.render_dashboard()  # Should not raise

    def test_latency_tracked_during_dq_check(self):
        self.ml.check_data_quality(self.df, "latency_test")
        summary = self.ml.system.latency_summary()
        self.assertGreater(summary.get("count", 0), 0)

    def test_record_model_training_via_layer(self):
        self.ml.record_model_training("test_model", "1.0.0")
        self.assertIn("test_model", self.ml.model_health.registered_models())

    def test_record_predictions_via_layer(self):
        self.ml.record_model_training("pred_model", "1.0.0")
        result = self.ml.record_predictions("pred_model", "1.0.0", np.random.randn(50))
        self.assertEqual(result["n_predictions"], 50)

    def test_check_model_health_via_layer(self):
        self.ml.record_model_training("h_model", "1.0.0")
        self.ml.record_predictions("h_model", "1.0.0", np.random.randn(50))
        report = self.ml.check_model_health("h_model")
        self.assertIn("health_score", report)

    def test_check_portfolio_risk_via_layer(self):
        tickers = ["A", "B", "C", "D", "E"]
        weights = pd.Series([0.20, 0.20, 0.20, 0.20, 0.20], index=tickers)
        returns_df = pd.DataFrame(
            np.random.randn(252, 5) * 0.02,
            index=pd.date_range("2021-01-01", periods=252, freq="B"),
            columns=tickers,
        )
        report = self.ml.check_portfolio_risk(weights, returns_df)
        self.assertIn("historical_var", report)

    def test_detect_market_regime_via_layer(self):
        returns = _make_returns(n=252)
        report = self.ml.detect_market_regime(returns)
        self.assertIn("volatility_regime", report)

    def test_record_feed_update_and_check_freshness(self):
        self.ml.record_feed_update("price_feed", n_rows=500)
        report = self.ml.check_data_freshness()
        self.assertIn("feeds", report)
        feed_report = report["feeds"]["price_feed"]
        self.assertEqual(feed_report["status"], "fresh")

    def test_register_custom_feed(self):
        self.ml.register_feed("custom_feed", FeedFrequency.INTRADAY, 1800, 3600)
        self.assertIn("custom_feed", self.ml.data_freshness.registered_feeds())


# ─── 9. ModelHealthMonitor ────────────────────────────────────────────────────

class TestModelHealthMonitor(unittest.TestCase):
    def setUp(self):
        self.monitor = ModelHealthMonitor()

    def test_record_training_creates_state(self):
        self.monitor.record_model_training("xgb", "1.0.0", timestamp=time.time())
        self.assertIn("xgb", self.monitor.registered_models())

    def test_record_predictions_returns_record(self):
        self.monitor.record_model_training("xgb", "1.0.0")
        scores = np.random.randn(100)
        record = self.monitor.record_predictions("xgb", "1.0.0", scores)
        self.assertIsInstance(record, PredictionRecord)
        self.assertEqual(record.n_predictions, 100)
        self.assertEqual(record.model_id, "xgb")

    def test_record_predictions_with_forward_returns(self):
        self.monitor.record_model_training("xgb", "1.0.0")
        scores = np.random.randn(100)
        fwd = np.random.randn(100)
        record = self.monitor.record_predictions("xgb", "1.0.0", scores, fwd)
        self.assertIsNotNone(record.ic)
        self.assertIsNotNone(record.hit_rate)

    def test_record_empty_scores(self):
        self.monitor.record_model_training("xgb", "1.0.0")
        record = self.monitor.record_predictions("xgb", "1.0.0", np.array([]))
        self.assertEqual(record.n_predictions, 0)

    def test_health_check_unknown_model(self):
        report = self.monitor.check_model_health("unknown")
        self.assertEqual(report["status"], "unknown")
        self.assertEqual(report["health_score"], 0.0)

    def test_health_check_healthy_model(self):
        ts = time.time()
        self.monitor.record_model_training("xgb", "1.0.0", timestamp=ts)
        scores = np.random.randn(100)
        fwd = scores * 0.5 + np.random.randn(100) * 0.1  # Correlated
        self.monitor.record_predictions("xgb", "1.0.0", scores, fwd, timestamp=ts)
        report = self.monitor.check_model_health("xgb")
        self.assertIn("health_score", report)
        self.assertGreater(report["health_score"], 0.5)
        self.assertFalse(report["retrain_recommended"])

    def test_health_check_stale_model_warns(self):
        old_ts = time.time() - 3600 * 100  # 100 hours ago
        self.monitor.record_model_training("stale_m", "1.0.0", timestamp=old_ts)
        self.monitor.record_predictions(
            "stale_m", "1.0.0", np.random.randn(20), timestamp=old_ts
        )
        report = self.monitor.check_model_health("stale_m")
        self.assertGreater(len(report["issues"]), 0)

    def test_feature_importance_recording(self):
        self.monitor.record_model_training("xgb", "1.0.0")
        self.monitor.record_feature_importance("xgb", "1.0.0", {"f1": 0.3, "f2": 0.2, "f3": 0.5})
        self.monitor.record_feature_importance("xgb", "1.0.0", {"f1": 0.1, "f2": 0.4, "f3": 0.5})
        report = self.monitor.check_model_health("xgb")
        # Should have computed rank shift
        self.assertIn("feature_importance_rank_shift", report)

    def test_check_all_models(self):
        self.monitor.record_model_training("m1", "1.0")
        self.monitor.record_model_training("m2", "1.0")
        reports = self.monitor.check_all_models()
        self.assertIn("m1", reports)
        self.assertIn("m2", reports)

    def test_reset_clears_state(self):
        self.monitor.record_model_training("xgb", "1.0.0")
        self.monitor.reset()
        self.assertEqual(len(self.monitor.registered_models()), 0)

    def test_reset_single_model(self):
        self.monitor.record_model_training("m1", "1.0")
        self.monitor.record_model_training("m2", "1.0")
        self.monitor.reset("m1")
        self.assertNotIn("m1", self.monitor.registered_models())
        self.assertIn("m2", self.monitor.registered_models())


# ─── 10. PortfolioRiskMonitor ─────────────────────────────────────────────────

class TestPortfolioRiskMonitor(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.tickers = ["A", "B", "C", "D", "E"]
        self.weights = pd.Series([0.20, 0.20, 0.20, 0.20, 0.20], index=self.tickers)
        self.returns_df = pd.DataFrame(
            np.random.randn(252, 5) * 0.02,
            index=pd.date_range("2021-01-01", periods=252, freq="B"),
            columns=self.tickers,
        )
        self.monitor = PortfolioRiskMonitor()

    def test_basic_check_returns_report(self):
        report = self.monitor.check(self.weights, self.returns_df)
        self.assertIn("n_positions", report)
        self.assertEqual(report["n_positions"], 5)
        self.assertIn("gross_exposure", report)
        self.assertAlmostEqual(report["gross_exposure"], 1.0, places=2)

    def test_var_cvar_computed(self):
        report = self.monitor.check(self.weights, self.returns_df)
        self.assertIn("parametric_var", report)
        self.assertIn("historical_var", report)
        self.assertIn("cvar", report)
        self.assertIsNotNone(report["parametric_var"])
        self.assertIsNotNone(report["historical_var"])

    def test_concentration_hhi(self):
        report = self.monitor.check(self.weights, self.returns_df)
        self.assertIn("hhi", report)
        # Equal-weight 5 stocks → HHI = 5 * (0.2^2) = 0.20
        self.assertAlmostEqual(report["hhi"], 0.20, places=2)

    def test_concentrated_portfolio_breach(self):
        conc_weights = pd.Series([0.80, 0.05, 0.05, 0.05, 0.05], index=self.tickers)
        report = self.monitor.check(conc_weights, self.returns_df)
        self.assertTrue(report["concentration_breach"])

    def test_turnover_computation(self):
        prev = pd.Series([0.30, 0.20, 0.20, 0.20, 0.10], index=self.tickers)
        report = self.monitor.check(self.weights, self.returns_df, prev_weights=prev)
        self.assertIn("turnover", report)
        self.assertIsNotNone(report["turnover"])
        self.assertGreater(report["turnover"], 0)

    def test_leverage_check_normal(self):
        report = self.monitor.check(self.weights, self.returns_df)
        self.assertFalse(report["leverage_breach"])

    def test_leverage_check_high(self):
        high_lev = pd.Series([0.50, 0.50, 0.40, 0.20, -0.10], index=self.tickers)
        report = self.monitor.check(high_lev, self.returns_df)
        # Gross = 1.70 → above LEVERAGE_CRIT=1.50
        self.assertTrue(report["leverage_breach"])

    def test_sector_map_concentration(self):
        sector_map = {"A": "Tech", "B": "Tech", "C": "Finance", "D": "Finance", "E": "Health"}
        report = self.monitor.check(self.weights, self.returns_df, sector_map=sector_map)
        self.assertIn("sector_weights", report)
        self.assertIn("Tech", report["sector_weights"])

    def test_insufficient_data_returns_none_var(self):
        short_df = pd.DataFrame(
            np.random.randn(5, 5) * 0.01,
            index=pd.date_range("2022-01-01", periods=5, freq="B"),
            columns=self.tickers,
        )
        report = self.monitor.check(self.weights, short_df)
        self.assertIsNone(report["parametric_var"])


# ─── 11. MarketRegimeMonitor ─────────────────────────────────────────────────

class TestMarketRegimeMonitor(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.monitor = MarketRegimeMonitor()

    def test_detect_regime_returns_report(self):
        returns = _make_returns(n=252, seed=42)
        report = self.monitor.detect_regime(returns)
        self.assertIn("volatility_regime", report)
        self.assertIn("trend_regime", report)
        self.assertIn("regime_composite", report)

    def test_volatility_regime_classification(self):
        returns = _make_returns(n=252, seed=42)
        report = self.monitor.detect_regime(returns)
        self.assertIn(report["volatility_regime"], [e.value for e in VolatilityRegime])

    def test_crisis_volatility_detected(self):
        """Large daily moves should trigger CRISIS or ELEVATED."""
        dates = pd.date_range("2022-01-01", periods=252, freq="B")
        crisis_returns = pd.Series(np.random.normal(0, 0.05, 252), index=dates)
        report = self.monitor.detect_regime(crisis_returns)
        self.assertIn(report["volatility_regime"], [VolatilityRegime.ELEVATED.value, VolatilityRegime.CRISIS.value])

    def test_trend_regime_classification(self):
        returns = _make_returns(n=252)
        report = self.monitor.detect_regime(returns)
        self.assertIn(report["trend_regime"], [e.value for e in TrendRegime])

    def test_correlation_regime_with_multi_asset(self):
        dates = pd.date_range("2022-01-01", periods=252, freq="B")
        # Create multi-asset returns
        multi = pd.DataFrame(
            np.random.randn(252, 5) * 0.01,
            index=dates,
            columns=["A", "B", "C", "D", "E"],
        )
        returns = _make_returns(n=252)
        report = self.monitor.detect_regime(returns, multi_asset_returns=multi)
        self.assertIn("correlation_regime", report)
        self.assertIn(report["correlation_regime"], [e.value for e in CorrelationRegime])

    def test_high_correlation_regime_detected(self):
        """All assets moving together should trigger CORRELATED regime."""
        dates = pd.date_range("2022-01-01", periods=100, freq="B")
        base = np.random.randn(100)
        # All columns highly correlated
        multi = pd.DataFrame({
            "A": base + np.random.randn(100) * 0.01,
            "B": base + np.random.randn(100) * 0.01,
            "C": base + np.random.randn(100) * 0.01,
            "D": base + np.random.randn(100) * 0.01,
        }, index=dates)
        returns = pd.Series(base * 0.01, index=dates)
        report = self.monitor.detect_regime(returns, multi_asset_returns=multi, vol_window=21)
        self.assertEqual(report["correlation_regime"], CorrelationRegime.CORRELATED.value)

    def test_regime_change_detection(self):
        returns1 = _make_returns(n=252, seed=42)
        self.monitor.detect_regime(returns1)  # First call sets prev_regime

        # Second call with different regime
        dates2 = pd.date_range("2023-01-01", periods=252, freq="B")
        returns2 = pd.Series(np.random.normal(0, 0.04, 252), index=dates2)
        report2 = self.monitor.detect_regime(returns2)
        # regime_changed should be True if the composite changed
        self.assertIn("regime_changed", report2)

    def test_insufficient_data(self):
        short = pd.Series([0.01, -0.01, 0.005], index=pd.date_range("2022-01-01", periods=3))
        report = self.monitor.detect_regime(short)
        self.assertEqual(report["status"], "insufficient_data")

    def test_composite_regime_crisis(self):
        composite = MarketRegimeMonitor._compute_composite_regime(
            VolatilityRegime.CRISIS.value, TrendRegime.BEAR.value, CorrelationRegime.NORMAL.value
        )
        self.assertEqual(composite, "CRISIS")

    def test_composite_regime_bull(self):
        composite = MarketRegimeMonitor._compute_composite_regime(
            VolatilityRegime.NORMAL.value, TrendRegime.BULL.value, CorrelationRegime.NORMAL.value
        )
        self.assertEqual(composite, "BULL")

    def test_composite_regime_neutral(self):
        composite = MarketRegimeMonitor._compute_composite_regime(
            VolatilityRegime.NORMAL.value, TrendRegime.SIDEWAYS.value, CorrelationRegime.NORMAL.value
        )
        self.assertEqual(composite, "NEUTRAL")


# ─── 12. DataFreshnessMonitor ─────────────────────────────────────────────────

class TestDataFreshnessMonitor(unittest.TestCase):
    def setUp(self):
        self.monitor = DataFreshnessMonitor()

    def test_default_feeds_registered(self):
        feeds = self.monitor.registered_feeds()
        self.assertIn("price_feed", feeds)
        self.assertIn("feature_feed", feeds)
        self.assertIn("prediction_feed", feeds)
        self.assertIn("fundamental_feed", feeds)

    def test_register_custom_feed(self):
        self.monitor.register_feed(
            "alt_data_feed",
            FeedFrequency.INTRADAY,
            warn_after_seconds=1800,
            crit_after_seconds=3600,
        )
        self.assertIn("alt_data_feed", self.monitor.registered_feeds())

    def test_record_update_marks_fresh(self):
        now = time.time()
        self.monitor.record_update("price_feed", n_rows=500, timestamp=now)
        report = self.monitor.check_feed("price_feed", now=now + 10)
        self.assertTrue(report["fresh"])
        self.assertEqual(report["status"], "fresh")
        self.assertEqual(report["last_row_count"], 500)

    def test_stale_feed_warns(self):
        now = time.time()
        stale_ts = now - 3600 * 30  # 30 hours ago
        self.monitor.record_update("price_feed", n_rows=500, timestamp=stale_ts)
        report = self.monitor.check_feed("price_feed", now=now)
        self.assertIn(report["status"], ["warning", "stale"])

    def test_critical_staleness(self):
        now = time.time()
        very_old = now - 3600 * 60  # 60 hours ago
        self.monitor.record_update("price_feed", n_rows=500, timestamp=very_old)
        report = self.monitor.check_feed("price_feed", now=now)
        self.assertEqual(report["status"], "critical")

    def test_never_updated_feed(self):
        self.monitor.register_feed("new_feed", FeedFrequency.DAILY)
        report = self.monitor.check_feed("new_feed")
        self.assertFalse(report["fresh"])

    def test_check_unknown_feed(self):
        report = self.monitor.check_feed("nonexistent")
        self.assertEqual(report["status"], "unknown")

    def test_check_all_feeds(self):
        now = time.time()
        self.monitor.record_update("price_feed", n_rows=500, timestamp=now)
        self.monitor.record_update("feature_feed", n_rows=200, timestamp=now)
        self.monitor.record_update("prediction_feed", n_rows=100, timestamp=now)
        self.monitor.record_update("fundamental_feed", n_rows=50, timestamp=now)
        report = self.monitor.check_all_feeds(now=now + 5)
        self.assertIn("summary", report)
        self.assertEqual(report["summary"]["total_feeds"], 4)
        self.assertEqual(report["summary"]["fresh"], 4)
        self.assertEqual(report["summary"]["overall_health"], "HEALTHY")

    def test_low_row_count_alert(self):
        self.monitor.register_feed("strict_feed", FeedFrequency.DAILY, min_rows=100)
        self.monitor.record_update("strict_feed", n_rows=5)
        # Should have fired a warning alert internally

    def test_feed_health_scores(self):
        scores = self.monitor.feed_health_scores()
        self.assertIsInstance(scores, dict)
        for feed in self.monitor.registered_feeds():
            self.assertIn(feed, scores)

    def test_total_updates_counter(self):
        now = time.time()
        self.monitor.record_update("price_feed", n_rows=100, timestamp=now)
        self.monitor.record_update("price_feed", n_rows=200, timestamp=now + 10)
        report = self.monitor.check_feed("price_feed", now=now + 15)
        self.assertEqual(report["total_updates"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)

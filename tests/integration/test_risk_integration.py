"""
tests/integration/test_risk_integration.py
─────────────────────────────────────────────
Integration tests verifying Phase 9 Risk Engine 2.0 interaction with
Portfolio Engine, Confidence Engine, and Monitoring Platform.
"""

import unittest
import numpy as np
import pandas as pd

from risk_layer.engine import InstitutionalRiskEngine
from portfolio_layer.optimizer import PortfolioOptimizer
from monitoring_layer.monitor import MonitoringLayer


class TestRiskIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
        np.random.seed(42)
        cls.dates = pd.date_range("2024-01-01", periods=150, freq="B")

        cls.returns_df = pd.DataFrame(
            np.random.normal(0.0005, 0.014, size=(150, 5)),
            index=cls.dates,
            columns=cls.tickers,
        )
        cls.adv_data = pd.Series([5e8, 3e8, 4e8, 6e8, 4.5e8], index=cls.tickers)
        cls.sector_map = {
            "RELIANCE.NS": "Energy",
            "TCS.NS": "Tech",
            "INFY.NS": "Tech",
            "HDFCBANK.NS": "Finance",
            "ICICIBANK.NS": "Finance",
        }
        cls.benchmark_returns = pd.Series(np.random.normal(0.0004, 0.01, size=150), index=cls.dates)

    def setUp(self):
        self.optimizer = PortfolioOptimizer()
        self.risk_engine = InstitutionalRiskEngine()
        self.monitoring = MonitoringLayer()

    def test_portfolio_engine_to_risk_engine_flow(self):
        """Verify portfolio optimization weights flow cleanly into InstitutionalRiskEngine audit."""
        for opt_name in ["equal_weight", "hrp", "risk_parity", "confidence_weighted"]:
            weights = self.optimizer.optimize(
                selected_tickers=set(self.tickers),
                optimizer_name=opt_name,
                returns_df=self.returns_df,
            )

            report = self.risk_engine.audit_portfolio_risk(
                weights=weights,
                returns_df=self.returns_df,
                adv_data=self.adv_data,
                sector_map=self.sector_map,
                benchmark_returns=self.benchmark_returns,
            )

            self.assertIsNotNone(report)
            self.assertGreater(report.var_95_historical, 0.0)
            self.assertGreater(report.cvar_95, 0.0)
            self.assertIsInstance(report.position_limits_passed, bool)

    def test_risk_engine_to_monitoring_platform_integration(self):
        """Verify risk metrics audit registers cleanly with Monitoring Platform health checks."""
        weights = self.optimizer.optimize(
            selected_tickers=set(self.tickers),
            optimizer_name="hrp",
            returns_df=self.returns_df,
        )

        risk_report = self.risk_engine.audit_portfolio_risk(
            weights=weights,
            returns_df=self.returns_df,
            adv_data=self.adv_data,
        )

        monitoring_report = self.monitoring.full_health_check(
            portfolio_weights=weights,
            returns_df=self.returns_df,
        )

        self.assertIn("overall_health", monitoring_report)
        self.assertIn("portfolio_risk", monitoring_report)
        self.assertGreater(monitoring_report["portfolio_risk"]["historical_var"], 0.0)


if __name__ == "__main__":
    unittest.main()

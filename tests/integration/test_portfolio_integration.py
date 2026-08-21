"""
tests/integration/test_portfolio_integration.py
─────────────────────────────────────────────────
Integration tests verifying Phase 8 Portfolio Engine 2.0 interaction with
Feature Store, Confidence Engine, Risk Engine, and Monitoring Platform.
"""

import unittest
import numpy as np
import pandas as pd

from portfolio_layer.optimizer import PortfolioOptimizer
from portfolio_layer.comparison import PortfolioComparisonSuite
from execution_layer.backtester import Backtester
from monitoring_layer.monitor import MonitoringLayer


class TestPortfolioIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
        np.random.seed(123)
        cls.dates = pd.date_range("2024-01-01", periods=150, freq="B")

        # Returns panel simulating Feature Store / Price data
        cls.returns_df = pd.DataFrame(
            np.random.normal(0.0006, 0.012, size=(150, 5)),
            index=cls.dates,
            columns=cls.tickers,
        )
        cls.scores_df = pd.DataFrame(
            np.random.normal(0.02, 0.05, size=(150, 5)),
            index=cls.dates,
            columns=cls.tickers,
        )
        cls.adv_df = pd.DataFrame(
            5e8,
            index=cls.dates,
            columns=cls.tickers,
        )

        # Confidence Engine outputs
        cls.confidence_df = pd.DataFrame(
            {
                "confidence_tier": ["HIGH", "HIGH", "MEDIUM", "LOW", "MEDIUM"],
                "prediction_std": [0.01, 0.015, 0.03, 0.07, 0.04],
            },
            index=cls.tickers,
        )

    def setUp(self):
        self.optimizer = PortfolioOptimizer()
        self.monitoring = MonitoringLayer()

    def test_portfolio_optimizer_with_confidence_and_returns(self):
        """Verify dynamic optimization using confidence tiers and returns panel."""
        for opt_name in ["equal_weight", "hrp", "risk_parity", "min_variance", "confidence_weighted", "kelly", "volatility_targeting"]:
            weights = self.optimizer.optimize(
                selected_tickers=set(self.tickers),
                optimizer_name=opt_name,
                returns_df=self.returns_df,
                alpha_scores=self.scores_df.iloc[-1],
                confidence_df=self.confidence_df,
                target_volatility=0.14,
            )
            self.assertEqual(len(weights), 5)
            self.assertAlmostEqual(weights.sum(), 1.0, places=4)

    def test_portfolio_comparison_suite_end_to_end(self):
        """Verify full execution of PortfolioComparisonSuite end-to-end."""
        suite = PortfolioComparisonSuite(initial_capital=100000.0, target_volatility=0.14)
        df_comp = suite.run_comparison(
            scores_df=self.scores_df,
            stock_returns=self.returns_df,
            confidence_df=self.confidence_df,
            adv_data=self.adv_df,
        )
        self.assertFalse(df_comp.empty)
        self.assertIn("cagr", df_comp.columns)
        self.assertIn("sharpe_ratio", df_comp.columns)
        self.assertIn("turnover", df_comp.columns)

    def test_monitoring_integration_with_portfolio_weights(self):
        """Verify portfolio weights risk metrics registration in Monitoring Platform."""
        weights = self.optimizer.optimize(
            selected_tickers=set(self.tickers),
            optimizer_name="hrp",
            returns_df=self.returns_df,
        )
        # Register in monitoring system
        report = self.monitoring.full_health_check(
            portfolio_weights=weights,
            returns_df=self.returns_df,
        )
        self.assertIn("overall_health", report)
        self.assertIn("portfolio_risk", report)


if __name__ == "__main__":
    unittest.main()

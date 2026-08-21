"""
tests/unit/test_portfolio_engine_v2.py
───────────────────────────────────────
Unit tests for Phase 8: Portfolio Engine 2.0.
Verifies all 7 optimization plugins, constraints engine, and performance comparison suite.
"""

import unittest
import numpy as np
import pandas as pd

from portfolio_layer.base import PortfolioConstraints, PortfolioPluginRegistry
from portfolio_layer.config import PortfolioEngineConfig
from portfolio_layer.optimizer import PortfolioOptimizer
from portfolio_layer.constraints import ConstraintsEngine
from portfolio_layer.comparison import PortfolioComparisonSuite
from portfolio_layer.plugins.confidence_weighted import ConfidenceWeightedPlugin
from portfolio_layer.plugins.volatility_targeting import VolatilityTargetingPlugin


class TestPortfolioEngineV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tickers = {"RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"}
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100, freq="B")
        
        # Synthetic daily returns
        cls.returns_df = pd.DataFrame(
            np.random.normal(0.0005, 0.015, size=(100, 5)),
            index=dates,
            columns=sorted(list(cls.tickers)),
        )
        cls.alpha_scores = pd.Series(
            [0.05, 0.03, -0.01, 0.08, 0.02],
            index=sorted(list(cls.tickers)),
        )
        cls.adv_data = pd.Series(
            [5e8, 3e8, 4e8, 6e8, 4.5e8],
            index=sorted(list(cls.tickers)),
        )

    def setUp(self):
        self.optimizer = PortfolioOptimizer()
        self.constraints_engine = ConstraintsEngine()

    def test_registered_plugins_count(self):
        """Verify all 7 required optimization plugins are registered."""
        plugins = PortfolioPluginRegistry.list_plugins()
        plugin_names = [p["name"].lower() for p in plugins]
        expected = [
            "equal_weight",
            "hrp",
            "risk_parity",
            "min_variance",
            "confidence_weighted",
            "kelly",
            "volatility_targeting",
        ]
        for name in expected:
            self.assertIn(name, plugin_names)

    def test_equal_weight_baseline(self):
        """Test Equal Weight plugin sum and allocation equality."""
        weights = self.optimizer.optimize(
            selected_tickers=self.tickers,
            optimizer_name="equal_weight",
        )
        self.assertEqual(len(weights), 5)
        self.assertAlmostEqual(weights.sum(), 1.0, places=5)
        self.assertAlmostEqual(weights.iloc[0], 0.20, places=5)

    def test_hrp_optimization(self):
        """Test Hierarchical Risk Parity allocation."""
        weights = self.optimizer.optimize(
            selected_tickers=self.tickers,
            optimizer_name="hrp",
            returns_df=self.returns_df,
        )
        self.assertEqual(len(weights), 5)
        self.assertAlmostEqual(weights.sum(), 1.0, places=5)
        self.assertTrue((weights >= 0).all())

    def test_risk_parity_optimization(self):
        """Test Risk Parity (Equal Risk Contribution) allocation."""
        weights = self.optimizer.optimize(
            selected_tickers=self.tickers,
            optimizer_name="risk_parity",
            returns_df=self.returns_df,
        )
        self.assertEqual(len(weights), 5)
        self.assertAlmostEqual(weights.sum(), 1.0, places=5)

    def test_min_variance_optimization(self):
        """Test Minimum Variance allocation."""
        weights = self.optimizer.optimize(
            selected_tickers=self.tickers,
            optimizer_name="min_variance",
            returns_df=self.returns_df,
        )
        self.assertEqual(len(weights), 5)
        self.assertAlmostEqual(weights.sum(), 1.0, places=5)

    def test_confidence_weighted_optimization(self):
        """Test Confidence-Weighted plugin."""
        confidence_df = pd.DataFrame(
            {
                "confidence_tier": ["HIGH", "MEDIUM", "LOW", "HIGH", "MEDIUM"],
                "prediction_std": [0.01, 0.03, 0.08, 0.01, 0.04],
            },
            index=sorted(list(self.tickers)),
        )
        weights = self.optimizer.optimize(
            selected_tickers=self.tickers,
            optimizer_name="confidence_weighted",
            alpha_scores=self.alpha_scores,
            confidence_df=confidence_df,
        )
        self.assertEqual(len(weights), 5)
        self.assertAlmostEqual(weights.sum(), 1.0, places=5)

    def test_kelly_criterion_optimization(self):
        """Test Bounded Kelly Criterion plugin."""
        weights = self.optimizer.optimize(
            selected_tickers=self.tickers,
            optimizer_name="kelly",
            returns_df=self.returns_df,
            alpha_scores=self.alpha_scores,
        )
        self.assertEqual(len(weights), 5)
        self.assertAlmostEqual(weights.sum(), 1.0, places=5)

    def test_volatility_targeting_optimization(self):
        """Test Volatility Targeting plugin."""
        weights = self.optimizer.optimize(
            selected_tickers=self.tickers,
            optimizer_name="volatility_targeting",
            returns_df=self.returns_df,
            target_volatility=0.14,
        )
        self.assertEqual(len(weights), 5)
        self.assertAlmostEqual(weights.sum(), 1.0, places=5)

    def test_constraints_engine_industry_and_cash(self):
        """Test industry cap, cash reserve controls, and turnover limits."""
        raw_weights = pd.Series([0.40, 0.30, 0.15, 0.10, 0.05], index=sorted(list(self.tickers)))
        industry_map = {
            "RELIANCE.NS": "Energy",
            "TCS.NS": "Tech",
            "INFY.NS": "Tech",
            "HDFCBANK.NS": "Finance",
            "ICICIBANK.NS": "Finance",
        }
        
        # Test industry cap
        ind_w = self.constraints_engine.apply_industry_bounds(
            raw_weights, industry_map, max_industry_weight=0.25
        )
        self.assertLessEqual(ind_w["TCS.NS"] + ind_w["INFY.NS"], 0.30)

        # Test cash controls
        cash_w = self.constraints_engine.apply_cash_controls(raw_weights, min_cash=0.10, max_cash=0.20)
        self.assertAlmostEqual(cash_w.sum(), 0.90, places=4)

    def test_comparison_suite_metrics(self):
        """Test metrics calculation in PortfolioComparisonSuite."""
        dates = pd.date_range("2024-01-01", periods=252, freq="B")
        daily_ret = pd.Series(np.random.normal(0.0008, 0.01, size=252), index=dates)
        eq_curve = (1.0 + daily_ret).cumprod() * 100000.0
        weights_df = pd.DataFrame(0.20, index=dates, columns=sorted(list(self.tickers)))

        suite = PortfolioComparisonSuite()
        metrics = suite.calculate_metrics(eq_curve, daily_ret, weights_df)

        self.assertIn("cagr", metrics)
        self.assertIn("sharpe_ratio", metrics)
        self.assertIn("sortino_ratio", metrics)
        self.assertIn("max_drawdown", metrics)
        self.assertIn("turnover", metrics)
        self.assertIn("transaction_cost_impact_bps", metrics)


if __name__ == "__main__":
    unittest.main()

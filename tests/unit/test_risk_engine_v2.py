"""
tests/unit/test_risk_engine_v2.py
───────────────────────────────────
Unit tests for Phase 9: Institutional Quant Risk Engine 2.0.
Verifies VaR/CVaR, Stress Testing, Scenario Analysis, Liquidity, Concentration,
Sector/Factor/Beta Exposure, Risk Heatmaps, Limits, Warnings, and 5 Formal Reports.
"""

import os
import json
import unittest
import numpy as np
import pandas as pd
from pathlib import Path

from risk_layer.config import RiskConfig
from risk_layer.engine import InstitutionalRiskEngine
from risk_layer.var_cvar import VaRCVaREngine
from risk_layer.limits import LimitsAuditEngine
from risk_layer.heatmaps import RiskHeatmapEngine
from risk_layer.factor_risk import FactorRiskEngine


class TestRiskEngineV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=252, freq="B")

        cls.returns_df = pd.DataFrame(
            np.random.normal(0.0005, 0.015, size=(252, 5)),
            index=dates,
            columns=cls.tickers,
        )
        cls.weights = pd.Series([0.30, 0.25, 0.20, 0.15, 0.10], index=cls.tickers)
        cls.adv_data = pd.Series([5e8, 3e8, 4e8, 6e8, 4.5e8], index=cls.tickers)
        cls.sector_map = {
            "RELIANCE.NS": "Energy",
            "TCS.NS": "Tech",
            "INFY.NS": "Tech",
            "HDFCBANK.NS": "Finance",
            "ICICIBANK.NS": "Finance",
        }
        cls.benchmark_returns = pd.Series(np.random.normal(0.0004, 0.01, size=252), index=dates)

    def setUp(self):
        self.config = RiskConfig()
        self.risk_engine = InstitutionalRiskEngine(self.config)

    def test_var_cvar_methodologies(self):
        """Test Historical, Parametric (Cornish-Fisher), and Monte Carlo VaR & CVaR."""
        var_eng = VaRCVaREngine()
        port_returns = (self.returns_df * self.weights).sum(axis=1)

        var95_h, cvar95_h = var_eng.historical_var_cvar(port_returns, confidence=0.95)
        var95_p, cvar95_p = var_eng.parametric_var_cvar(port_returns, confidence=0.95, use_cornish_fisher=True)
        var95_mc, cvar95_mc = var_eng.monte_carlo_var_cvar(port_returns, confidence=0.95, n_simulations=5000)

        self.assertGreater(var95_h, 0.0)
        self.assertGreaterEqual(cvar95_h, var95_h)
        self.assertGreater(var95_p, 0.0)
        self.assertGreater(var95_mc, 0.0)

    def test_factor_and_beta_exposures(self):
        """Test Factor exposure estimation and benchmark Beta calculation."""
        factor_eng = FactorRiskEngine()
        exposures = factor_eng.compute_factor_exposures(self.weights, returns_df=self.returns_df)
        beta = factor_eng.compute_portfolio_beta(self.weights, self.returns_df, self.benchmark_returns)

        self.assertIn("Momentum", exposures)
        self.assertIn("Volatility", exposures)
        self.assertIsInstance(beta, float)
        self.assertGreater(beta, 0.0)

    def test_concentration_and_limits_warnings(self):
        """Test concentration metrics and warning alert generation on limit breaches."""
        limits_eng = LimitsAuditEngine(self.config)
        
        # Normal weights audit
        passed, checks, warnings = limits_eng.audit_limits(self.weights, sector_map=self.sector_map)
        self.assertIsInstance(passed, bool)

        # Concentrated weights forcing single position and HHI breach
        concentrated_w = pd.Series([0.80, 0.05, 0.05, 0.05, 0.05], index=self.tickers)
        passed_c, checks_c, warnings_c = limits_eng.audit_limits(
            concentrated_w, sector_map=self.sector_map, var_95=0.08, cvar_95=0.12
        )
        self.assertFalse(passed_c)
        self.assertGreater(len(warnings_c), 0)

    def test_risk_heatmaps_engine(self):
        """Test RiskHeatmapEngine marginal risk contributions and correlation matrix."""
        heatmap_eng = RiskHeatmapEngine()
        heatmaps = heatmap_eng.compute_risk_heatmaps(self.weights, self.returns_df, self.sector_map)

        self.assertIn("portfolio_volatility", heatmaps)
        self.assertIn("asset_risk_attribution", heatmaps)
        self.assertIn("correlation_matrix", heatmaps)
        self.assertIn("sector_risk_contribution_pct", heatmaps)

    def test_full_institutional_risk_audit_and_5_reports(self):
        """Verify InstitutionalRiskEngine full audit and 5 formal report output files."""
        report = self.risk_engine.audit_portfolio_risk(
            weights=self.weights,
            returns_df=self.returns_df,
            adv_data=self.adv_data,
            sector_map=self.sector_map,
            benchmark_returns=self.benchmark_returns,
        )

        self.assertIsNotNone(report)
        self.assertGreater(report.var_95_historical, 0.0)
        self.assertGreater(report.cvar_95, 0.0)
        self.assertGreater(report.days_to_liquidate_95pct, 0.0)

        # Verify 5 formal institutional report files
        reports_dir = Path("reports")
        expected_files = [
            "risk_report.json",
            "exposure_report.json",
            "var_report.json",
            "stress_test_report.json",
            "liquidity_report.json",
        ]
        for fname in expected_files:
            fpath = reports_dir / fname
            self.assertTrue(fpath.exists(), f"Missing expected report file: {fname}")
            with open(fpath, "r") as f:
                content = json.load(f)
                self.assertIsInstance(content, dict)


if __name__ == "__main__":
    unittest.main()

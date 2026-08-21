"""
tests/integration/test_api_integration.py
────────────────────────────────────────
Integration tests verifying backend API interaction with ML, Risk, and Backtest engines.
"""

import unittest
from fastapi.testclient import TestClient
from backend_services.app import create_app


class TestAPIIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.test_client = TestClient(cls.app)
        cls.headers = {"X-API-Key": "qsx-secret-api-key-2026"}

    def test_health_status(self):
        """Test the health and system diagnostics endpoint."""
        response = self.test_client.get("/api/v2/health/status", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertIn("uptime_seconds", data)

    def test_backtest_run_integration(self):
        """Test running a full backtest via the API."""
        payload = {
            "initial_capital": 10000000.0,
            "transaction_cost_pct": 0.0015,
            "apply_vol_targeting": True,
            "signal_lag_days": 1,
            "rebalance_frequency": "monthly"
        }
        response = self.test_client.post(
            "/api/v2/backtest/run",
            json=payload,
            headers=self.headers
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("cagr", data)
        self.assertIn("sharpe_ratio", data)
        self.assertIn("max_drawdown", data)
        self.assertIn("final_equity", data)
        self.assertIn("equity_curve", data)
        self.assertIsInstance(data["equity_curve"], list)

    def test_risk_audit_integration(self):
        """Test the risk audit execution via API."""
        payload = {
            "confidence_level": 0.95,
            "lookback_days": 252,
            "include_stress_testing": True
        }
        response = self.test_client.post(
            "/api/v2/risk/audit",
            json=payload,
            headers=self.headers
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("var_95", data)
        self.assertIn("cvar_95", data)
        self.assertIn("tail_risk_ratio", data)
        self.assertIn("risk_grade", data)

    def test_analyst_explain_integration(self):
        """Test the AI Quant Analyst prediction explanation API."""
        payload = {
            "symbol": "RELIANCE",
            "predicted_score": 0.045,
            "probability": 0.78,
            "shap_values": {"mom_60": 0.03, "vol_20": -0.01}
        }
        response = self.test_client.post(
            "/api/v2/analyst/explain-prediction",
            json=payload,
            headers=self.headers
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["symbol"], "RELIANCE")
        self.assertIn("direction", data)
        self.assertIn("narrative", data)
        self.assertIn("key_drivers", data)


if __name__ == "__main__":
    unittest.main()

"""
tests/performance/locustfile.py
───────────────────────────────
Locust load testing scenarios for the FastAPI backend.
"""

from locust import HttpUser, task, between
import json

class QuantSphereXApiUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        """Setup user state or retrieve tokens if needed."""
        self.headers = {"X-API-Key": "test_key_123"}
        self.backtest_payload = {
            "initial_capital": 10000000.0,
            "transaction_cost_pct": 0.0015,
            "apply_vol_targeting": False,
            "signal_lag_days": 1,
            "rebalance_frequency": "monthly"
        }
        self.risk_payload = {
            "confidence_level": 0.95,
            "lookback_days": 252,
            "include_stress_testing": False
        }

    @task(3)
    def test_health(self):
        """High frequency lightweight endpoint."""
        with self.client.get("/api/v2/health/status", headers=self.headers, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with {response.status_code}")

    @task(1)
    def test_backtest_run(self):
        """Heavy compute endpoint."""
        with self.client.post("/api/v2/backtest/run", headers=self.headers, json=self.backtest_payload, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with {response.status_code}")

    @task(1)
    def test_risk_audit(self):
        """Medium compute endpoint."""
        with self.client.post("/api/v2/risk/audit", headers=self.headers, json=self.risk_payload, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with {response.status_code}")

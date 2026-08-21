"""
tests/conftest.py
─────────────────
Global Pytest fixtures and mock data generators for QuantSphereX testing.
Provides deterministic data for unit, integration, and regression testing.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any

# Ensure deterministic random generation across all tests
np.random.seed(42)

@pytest.fixture(scope="session")
def mock_stock_returns() -> pd.DataFrame:
    """Generates synthetic daily returns for 5 tickers over 1 year (252 days)."""
    dates = pd.date_range("2023-01-01", periods=252, freq="B")
    tickers = [f"TICKER_{i:02d}" for i in range(5)]
    np.random.seed(42) # Re-seed inside fixture for safety
    return pd.DataFrame(
        np.random.normal(0.0004, 0.011, (252, 5)),
        index=dates,
        columns=tickers
    )

@pytest.fixture(scope="session")
def mock_weights_schedule(mock_stock_returns: pd.DataFrame) -> pd.DataFrame:
    """Generates a monthly uniform weights schedule for the tickers."""
    dates = mock_stock_returns.index
    monthly = pd.date_range(dates[0], dates[-1], freq="MS")
    monthly = monthly[monthly.isin(dates)]
    
    if len(monthly) == 0:
        # Fallback if dates don't perfectly align with month-start
        monthly = dates[::21]

    tickers = mock_stock_returns.columns
    return pd.DataFrame(1.0 / len(tickers), index=monthly, columns=tickers)

@pytest.fixture(scope="session")
def mock_alpha_signals() -> pd.DataFrame:
    """Generates mock alpha predictions/signals for a single day."""
    return pd.DataFrame({
        "symbol": ["TICKER_00", "TICKER_01", "TICKER_02"],
        "score": [0.05, 0.02, -0.03],
        "probability": [0.85, 0.60, 0.70]
    })

@pytest.fixture(scope="session")
def test_client():
    """Provides a FastAPI TestClient for integration tests."""
    from fastapi.testclient import TestClient
    from backend_services.app import create_app
    app = create_app()
    return TestClient(app)

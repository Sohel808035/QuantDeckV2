"""
tests/performance/test_benchmarks.py
────────────────────────────────────
Micro-benchmarking of critical hot paths (e.g., Risk VaR calculations).
"""

import pytest
import numpy as np
import pandas as pd
from execution_layer.backtesting.metrics import MetricsEngine
from risk_layer.engine import InstitutionalRiskEngine, RiskConfig

def test_benchmark_metrics_engine(benchmark):
    """Benchmarks the speed of full-period metric calculations."""
    engine = MetricsEngine()
    
    # 5 years of daily data
    n_days = 252 * 5
    equity = pd.Series((1 + np.random.normal(0.0004, 0.01, n_days)).cumprod() * 10000)
    ret = equity.pct_change().fillna(0)
    turn = pd.Series(np.random.uniform(0.01, 0.05, n_days))
    fixed = pd.Series(0.0, index=ret.index)
    impact = pd.Series(0.0, index=ret.index)
    
    def run_metrics():
        return engine.full_period_metrics(equity, ret, turn, fixed, impact)

    result = benchmark(run_metrics)
    assert "cagr" in result

def test_benchmark_var_calculation(benchmark):
    """Benchmarks Historical VaR calculation in the Risk Engine."""
    config = RiskConfig()
    engine = InstitutionalRiskEngine(config)
    
    # Large synthetic returns panel for 100 assets over 2 years
    n_days = 504
    n_assets = 100
    returns = pd.DataFrame(np.random.normal(0, 0.015, (n_days, n_assets)))
    weights = pd.Series(1.0/n_assets, index=returns.columns)
    
    def run_var():
        return engine.compute_historical_var(returns, weights, confidence_level=0.95)

    var = benchmark(run_var)
    assert var > 0

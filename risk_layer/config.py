"""
risk_layer/config.py
────────────────────
Configuration DTO for the QuantSphereX Institutional Risk Engine.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class RiskConfig:
    """Master configuration for institutional risk evaluation, limits, and warning thresholds."""
    portfolio_value: float = 10_000_000.0  # 10 Million INR base portfolio
    confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99])
    historical_window_days: int = 504       # 2 years of daily returns
    monte_carlo_simulations: int = 10_000   # MC paths

    # VaR & Loss Limits
    max_var_95_pct: float = 0.03            # Max 3.0% daily VaR 95%
    max_cvar_95_pct: float = 0.05           # Max 5.0% daily CVaR 95%
    max_leverage: float = 1.50              # Max 150% gross exposure

    # Position & Concentration Limits
    max_single_position_pct: float = 0.10   # Max 10% per stock
    max_sector_exposure_pct: float = 0.30   # Max 30% per sector
    max_industry_exposure_pct: float = 0.15 # Max 15% per industry
    max_hhi_threshold: float = 0.15         # HHI concentration cap
    min_effective_n_stocks: float = 15.0    # Min N_eff = 1/sum(w^2)

    # Liquidity & Execution Limits
    max_adv_participation_pct: float = 0.05 # Max 5% daily volume participation
    max_days_to_liquidate: float = 3.0      # Must be liquidatable within 3 days

    # Tail & Stress
    evt_threshold_quantile: float = 0.05    # 5% left-tail quantile for EVT POT
    stress_max_drawdown_pct: float = 0.20   # 20% max stress loss threshold

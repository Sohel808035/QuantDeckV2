"""
portfolio_layer/plugins/volatility_targeting.py
──────────────────────────────────────────────────
Volatility Targeting Portfolio Plugin (Phase 8).
Scales asset weights dynamically to achieve a target annualized portfolio volatility.
"""

from __future__ import annotations
import logging
from typing import Set, Optional, Dict, Any
import pandas as pd
import numpy as np

from portfolio_layer.base import BasePortfolioPlugin, PortfolioPluginRegistry, PortfolioConstraints

logger = logging.getLogger(__name__)


@PortfolioPluginRegistry.register
class VolatilityTargetingPlugin(BasePortfolioPlugin):
    """
    Volatility Targeting Portfolio Plugin.
    Rescales allocation to enforce a target annualized portfolio volatility.
    """

    @property
    def name(self) -> str:
        return "volatility_targeting"

    @property
    def description(self) -> str:
        return "Scales allocation weights dynamically to target a fixed annualized portfolio volatility level."

    def optimize(
        self,
        selected_tickers: Set[str],
        returns_df: Optional[pd.DataFrame] = None,
        alpha_scores: Optional[pd.Series] = None,
        adv_data: Optional[pd.Series] = None,
        constraints: Optional[PortfolioConstraints] = None,
        target_volatility: float = 0.14,
        **kwargs,
    ) -> pd.Series:
        if not selected_tickers:
            return pd.Series(dtype=float)

        tickers = sorted(list(selected_tickers))

        # Baseline equal weight
        base_w = pd.Series(1.0 / len(tickers), index=tickers)
        if alpha_scores is not None and not alpha_scores.empty:
            scores = alpha_scores.reindex(tickers).fillna(0.0)
            pos_scores = np.maximum(scores, 0.0)
            if pos_scores.sum() > 0:
                base_w = pos_scores / pos_scores.sum()

        if returns_df is None or returns_df.empty:
            return base_w

        sub_ret = returns_df[[t for t in tickers if t in returns_df.columns]].dropna()
        if sub_ret.empty or len(sub_ret.columns) < 2:
            return base_w

        cov = sub_ret.cov() * 252.0
        w_vec = base_w.reindex(cov.index).fillna(0.0).values

        port_var = float(np.dot(w_vec, np.dot(cov.values, w_vec)))
        port_vol = float(np.sqrt(max(port_var, 1e-6)))

        scalar = target_volatility / port_vol if port_vol > 0 else 1.0
        # Bounded scaling [0.20, 1.50]
        scalar = float(np.clip(scalar, 0.20, 1.50))

        scaled_w = base_w * scalar
        return scaled_w / scaled_w.sum()

"""
portfolio_layer/plugins/confidence_weighted.py
────────────────────────────────────────────────
Confidence-Weighted Allocation Plugin (Phase 8).
Scales position sizes by combining prediction scores with prediction confidence tiers and variance.
"""

from __future__ import annotations
import logging
from typing import Set, Optional, Dict, Any
import pandas as pd
import numpy as np

from portfolio_layer.base import BasePortfolioPlugin, PortfolioPluginRegistry, PortfolioConstraints

logger = logging.getLogger(__name__)


@PortfolioPluginRegistry.register
class ConfidenceWeightedPlugin(BasePortfolioPlugin):
    """
    Confidence-Weighted Allocation Plugin.
    Modulates alpha scores by model confidence tiers and prediction uncertainty.
    """

    @property
    def name(self) -> str:
        return "confidence_weighted"

    @property
    def description(self) -> str:
        return "Scales asset allocation using prediction confidence tiers, ensemble variance, and alpha magnitude."

    def optimize(
        self,
        selected_tickers: Set[str],
        returns_df: Optional[pd.DataFrame] = None,
        alpha_scores: Optional[pd.Series] = None,
        adv_data: Optional[pd.Series] = None,
        constraints: Optional[PortfolioConstraints] = None,
        confidence_df: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> pd.Series:
        if not selected_tickers:
            return pd.Series(dtype=float)

        tickers = sorted(list(selected_tickers))

        if alpha_scores is None or alpha_scores.empty:
            logger.warning("[ConfidenceWeighted] No alpha_scores provided; falling back to Equal Weight.")
            return pd.Series(1.0 / len(tickers), index=tickers)

        scores = alpha_scores.reindex(tickers).fillna(0.0)

        # Baseline positive tilt
        pos_scores = np.maximum(scores, 0.0)
        if pos_scores.sum() == 0:
            pos_scores = pd.Series(1.0 / len(tickers), index=tickers)
        else:
            pos_scores = pos_scores / pos_scores.sum()

        # Apply confidence multipliers if confidence_df is provided
        confidence_multiplier = pd.Series(1.0, index=tickers)
        if confidence_df is not None and not confidence_df.empty:
            for t in tickers:
                if t in confidence_df.index:
                    row = confidence_df.loc[t]
                    tier = row.get("confidence_tier", "MEDIUM")
                    std = row.get("prediction_std", 0.05)

                    tier_scalar = 1.25 if tier == "HIGH" else (0.75 if tier == "LOW" else 1.0)
                    std_scalar = 1.0 / (1.0 + float(std) * 5.0) if pd.notna(std) else 1.0

                    confidence_multiplier[t] = tier_scalar * std_scalar

        weighted = pos_scores * confidence_multiplier
        total_w = weighted.sum()

        if total_w == 0:
            return pd.Series(1.0 / len(tickers), index=tickers)

        return weighted / total_w

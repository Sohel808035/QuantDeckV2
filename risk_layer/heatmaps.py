"""
risk_layer/heatmaps.py
───────────────────────
Phase 9: Portfolio Risk Heatmaps & Marginal Risk Contribution (MRC) Engine.
Calculates asset-level and sector-level risk contributions, Component VaR, and correlation heatmaps.
"""

from __future__ import annotations
import logging
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class RiskHeatmapEngine:
    """Computes Marginal Risk Contribution (MRC), Component VaR, and Risk Heatmap matrices."""

    def compute_risk_heatmaps(
        self,
        weights: pd.Series,
        returns_df: pd.DataFrame,
        sector_map: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Calculates:
          1. Marginal Risk Contribution (MRC) per asset
          2. Percentage Risk Contribution (RC%) per asset
          3. Component VaR (95%) per asset
          4. Asset correlation matrix for heatmap rendering
          5. Sector-level risk contribution breakdown
        """
        if weights.empty or returns_df.empty:
            return {}

        common_tickers = sorted(list(set(weights.index) & set(returns_df.columns)))
        if len(common_tickers) < 2:
            return {}

        w = weights.reindex(common_tickers).fillna(0.0)
        w = w / w.sum()
        w_vec = w.values

        sub_ret = returns_df[common_tickers].dropna()
        if len(sub_ret) < 5:
            return {}

        cov = sub_ret.cov() * 252.0  # Annualized covariance
        corr = sub_ret.corr().fillna(0.0)

        # Portfolio Volatility = sqrt(w^T * Cov * w)
        port_var = float(np.dot(w_vec, np.dot(cov.values, w_vec)))
        port_vol = float(np.sqrt(max(port_var, 1e-6)))

        # Marginal Risk Contribution (MRC) = (Cov * w) / port_vol
        mrc = np.dot(cov.values, w_vec) / port_vol

        # Component Risk Contribution (RC) = w_i * MRC_i
        rc = w_vec * mrc
        rc_pct = rc / port_vol if port_vol > 0 else np.zeros_like(rc)

        # Component VaR (95% 1-day)
        hist_var_95 = -float(np.percentile((sub_ret * w_vec).sum(axis=1), 5))
        c_var = w_vec * (np.dot(cov.values / 252.0, w_vec) / max(float(np.sqrt(port_var / 252.0)), 1e-6))

        df_risk = pd.DataFrame(
            {
                "weight": w.values,
                "marginal_risk_contribution": mrc,
                "risk_contribution": rc,
                "risk_contribution_pct": rc_pct,
                "component_var_95": c_var,
            },
            index=common_tickers,
        )

        # Sector risk breakdown
        sector_risk = {}
        if sector_map:
            df_risk["Sector"] = df_risk.index.map(sector_map).fillna("Other")
            sector_risk = df_risk.groupby("Sector")["risk_contribution_pct"].sum().to_dict()

        return {
            "portfolio_volatility": round(port_vol, 4),
            "asset_risk_attribution": df_risk.round(4).to_dict(orient="index"),
            "correlation_matrix": corr.round(4).to_dict(),
            "sector_risk_contribution_pct": {k: round(v, 4) for k, v in sector_risk.items()},
        }

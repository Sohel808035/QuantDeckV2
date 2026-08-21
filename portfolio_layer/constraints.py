"""
portfolio_layer/constraints.py
───────────────────────────────
QuantSphereX Portfolio Constraints Engine.
Applies asset caps, sector neutralization, beta targeting, and ADV liquidity filters.
"""

from __future__ import annotations
import logging
from typing import Dict, Optional, Set
import pandas as pd
import numpy as np

from portfolio_layer.base import PortfolioConstraints

logger = logging.getLogger(__name__)


class ConstraintsEngine:
    """Applies institutional constraints and filters to raw optimization weights."""

    def __init__(self, constraints: Optional[PortfolioConstraints] = None):
        self.constraints = constraints or PortfolioConstraints()

    def apply_all_constraints(
        self,
        weights: pd.Series,
        adv_data: Optional[pd.Series] = None,
        sector_map: Optional[Dict[str, str]] = None,
        benchmark_sector_weights: Optional[Dict[str, float]] = None,
        stock_betas: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Applies sequential constraint pipeline:
          1. Min/Max asset weight bounds (with iterative re-normalization)
          2. ADV Liquidity participation cap
          3. Sector allocation bounds
          4. Portfolio Beta target bounds
          5. Sum normalization to 1.0
        """
        if weights.empty:
            return weights

        w = weights.copy()

        # 1. ADV Liquidity participation cap
        if adv_data is not None and hasattr(adv_data, "reindex") and len(adv_data) > 0:
            adv_subset = adv_data.reindex(w.index).fillna(0)
            max_w = (adv_subset * self.constraints.max_adv_pct) / self.constraints.portfolio_value
            w = np.minimum(w, max_w)

        # 2. Sector allocation bounds
        if sector_map is not None and benchmark_sector_weights is not None:
            w = self.sector_neutralize(w, sector_map, benchmark_sector_weights)

        # 3. Portfolio Beta bounds
        if stock_betas is not None and len(stock_betas) > 0:
            w = self.beta_target(
                w, stock_betas, target_range=(self.constraints.target_beta_min, self.constraints.target_beta_max)
            )

        # 4. Min/Max asset weight bounds (iterative clipping)
        w = self._clip_and_normalize(
            w,
            min_w=self.constraints.min_weight_per_asset,
            max_w=self.constraints.max_weight_per_asset,
        )

        return w

    def _clip_and_normalize(
        self,
        weights: pd.Series,
        min_w: float,
        max_w: float,
        max_iter: int = 10,
    ) -> pd.Series:
        w = weights.copy()
        for _ in range(max_iter):
            w = w.clip(lower=min_w, upper=max_w)
            total = w.sum()
            if total == 0:
                return pd.Series(1.0 / len(w), index=w.index)
            if abs(total - 1.0) < 1e-5:
                break
            # Scale uncapped weights
            uncapped = (w > min_w) & (w < max_w)
            if not uncapped.any():
                w = w / total
                break
            excess = 1.0 - w[~uncapped].sum()
            if excess <= 0:
                w[uncapped] = 0.0
                break
            w[uncapped] = (w[uncapped] / w[uncapped].sum()) * excess
        return w

    def sector_neutralize(
        self,
        weights: pd.Series,
        sector_map: Dict[str, str],
        benchmark_weights: Dict[str, float],
        max_deviation: float = 0.05,
    ) -> pd.Series:
        """Adjusts weights to ensure sector allocations stay within benchmark +/- max_deviation."""
        if weights.empty:
            return weights

        df = pd.DataFrame({"weight": weights})
        df["Sector"] = df.index.map(sector_map).fillna("Other")
        port_sector_w = df.groupby("Sector")["weight"].sum()

        for sector, b_weight in benchmark_weights.items():
            current_w = port_sector_w.get(sector, 0.0)
            if current_w > 0:
                target_w = max(min(current_w, b_weight + max_deviation), b_weight - max_deviation)
                scalar = target_w / current_w
                df.loc[df["Sector"] == sector, "weight"] *= scalar

        result = df["weight"]
        if result.sum() > 0:
            result = result / result.sum()
        return result

        return w

    def apply_industry_bounds(
        self,
        weights: pd.Series,
        industry_map: Dict[str, str],
        max_industry_weight: Optional[float] = None,
        max_iter: int = 10,
    ) -> pd.Series:
        """Enforces maximum weight cap per industry with iterative normalization."""
        if weights.empty or not industry_map:
            return weights

        limit = max_industry_weight or self.constraints.max_industry_weight
        df = pd.DataFrame({"weight": weights.copy()})
        df["Industry"] = df.index.map(industry_map).fillna("Other")

        for _ in range(max_iter):
            ind_sums = df.groupby("Industry")["weight"].sum()
            breached = ind_sums[ind_sums > limit + 1e-6]
            if breached.empty:
                break
            for ind, ind_w in breached.items():
                df.loc[df["Industry"] == ind, "weight"] *= (limit / ind_w)

            uncapped_inds = ind_sums[ind_sums <= limit].index
            total_capped_w = df[df["Industry"].isin(breached.index)]["weight"].sum()
            rem_budget = max(0.0, 1.0 - total_capped_w)

            uncapped_mask = df["Industry"].isin(uncapped_inds)
            if uncapped_mask.any() and df.loc[uncapped_mask, "weight"].sum() > 0:
                df.loc[uncapped_mask, "weight"] = (
                    df.loc[uncapped_mask, "weight"] / df.loc[uncapped_mask, "weight"].sum()
                ) * rem_budget
            else:
                break

        res = df["weight"]
        if res.sum() > 0 and abs(res.sum() - 1.0) > 1e-5:
            res = res / res.sum()
        return res

    def apply_cash_controls(
        self,
        weights: pd.Series,
        min_cash: Optional[float] = None,
        max_cash: Optional[float] = None,
    ) -> pd.Series:
        """Applies cash allocation reserve bounds."""
        if weights.empty:
            return weights

        min_c = min_cash if min_cash is not None else self.constraints.min_cash_pct
        max_c = max_cash if max_cash is not None else self.constraints.max_cash_pct

        target_cash = max(min_c, min(max_c, self.constraints.min_cash_pct))
        invested_target = 1.0 - target_cash

        total_w = weights.sum()
        if total_w > 0:
            weights = (weights / total_w) * invested_target
        return weights

    def apply_turnover_limit(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        max_turnover: Optional[float] = None,
    ) -> pd.Series:
        """Clips portfolio rebalance trades to respect maximum turnover budget."""
        if current_weights.empty or target_weights.empty:
            return target_weights

        limit = max_turnover if max_turnover is not None else self.constraints.max_turnover
        union_idx = current_weights.index.union(target_weights.index)
        cur = current_weights.reindex(union_idx, fill_value=0.0)
        tgt = target_weights.reindex(union_idx, fill_value=0.0)

        trade = tgt - cur
        total_turnover = float(trade.abs().sum() / 2.0)

        if total_turnover > limit and total_turnover > 0:
            scalar = limit / total_turnover
            adjusted = cur + (trade * scalar)
            return adjusted[adjusted.abs() > 1e-6]

        return target_weights

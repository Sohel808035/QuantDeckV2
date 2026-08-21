"""
risk_layer/limits.py
────────────────────
Position & Concentration Limits Audit Engine for Phase 9.
Evaluates Herfindahl-Hirschman Index (HHI), Effective N, Sector/Industry caps, VaR/CVaR limits,
and generates automated warning alerts on any breach.
"""

from __future__ import annotations
import logging
from typing import Dict, Tuple, Optional, List, Any
import pandas as pd
import numpy as np

from risk_layer.config import RiskConfig

logger = logging.getLogger(__name__)


class LimitsAuditEngine:
    """Audits portfolio weights against regulatory and institutional position & concentration limits."""

    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()

    def concentration_metrics(self, weights: pd.Series) -> Dict[str, float]:
        """
        Computes Herfindahl-Hirschman Index (HHI), Effective Number of Assets (N_eff),
        Top-5, and Top-10 concentration ratios.
        """
        if weights.empty or weights.sum() == 0:
            return {"hhi_index": 0.0, "effective_n_stocks": 0.0, "top_5_concentration": 0.0, "top_10_concentration": 0.0}

        w = (weights / weights.sum()).values

        # 1. HHI = sum(w_i ^ 2)
        hhi = float(np.sum(w**2))

        # 2. Effective Number of Stocks N_eff = 1 / HHI
        n_eff = float(1.0 / hhi) if hhi > 0 else 0.0

        # 3. Top-5 and Top-10 concentration ratios
        sorted_w = np.sort(w)[::-1]
        top5 = float(np.sum(sorted_w[:5]))
        top10 = float(np.sum(sorted_w[:10]))

        return {
            "hhi_index": round(hhi, 4),
            "effective_n_stocks": round(n_eff, 2),
            "top_5_concentration": round(top5, 4),
            "top_10_concentration": round(top10, 4),
        }

    def audit_limits(
        self,
        weights: pd.Series,
        sector_map: Optional[Dict[str, str]] = None,
        industry_map: Optional[Dict[str, str]] = None,
        var_95: Optional[float] = None,
        cvar_95: Optional[float] = None,
        days_to_liquidate: Optional[float] = None,
    ) -> Tuple[bool, Dict[str, bool], List[Dict[str, Any]]]:
        """
        Audits portfolio weights against configured limit thresholds and returns warnings.
        Returns (all_limits_passed, checks_dict, warnings_list).
        """
        if weights.empty:
            return True, {}, []

        w = weights / weights.sum()
        max_pos = float(w.max())
        hhi_info = self.concentration_metrics(w)
        warnings: List[Dict[str, Any]] = []

        checks = {
            "single_position_limit": max_pos <= self.config.max_single_position_pct,
            "hhi_limit": hhi_info["hhi_index"] <= self.config.max_hhi_threshold,
            "effective_n_limit": hhi_info["effective_n_stocks"] >= self.config.min_effective_n_stocks,
        }

        if not checks["single_position_limit"]:
            top_ticker = w.idxmax()
            warnings.append({
                "severity": "WARNING",
                "category": "POSITION_LIMIT",
                "metric": "max_single_position_pct",
                "value": round(max_pos, 4),
                "threshold": self.config.max_single_position_pct,
                "message": f"Single position cap breached: '{top_ticker}' is {max_pos:.2%} (max {self.config.max_single_position_pct:.2%}).",
            })

        if not checks["hhi_limit"]:
            warnings.append({
                "severity": "WARNING",
                "category": "CONCENTRATION",
                "metric": "max_hhi_threshold",
                "value": round(hhi_info["hhi_index"], 4),
                "threshold": self.config.max_hhi_threshold,
                "message": f"HHI concentration index breached: {hhi_info['hhi_index']:.4f} > max {self.config.max_hhi_threshold:.4f}.",
            })

        if sector_map is not None:
            df = pd.DataFrame({"weight": w})
            df["Sector"] = df.index.map(sector_map).fillna("Other")
            sector_w = df.groupby("Sector")["weight"].sum()
            max_sec = float(sector_w.max())
            checks["sector_limit"] = max_sec <= self.config.max_sector_exposure_pct
            if not checks["sector_limit"]:
                top_sec = sector_w.idxmax()
                warnings.append({
                    "severity": "WARNING",
                    "category": "SECTOR_LIMIT",
                    "metric": "max_sector_exposure_pct",
                    "value": round(max_sec, 4),
                    "threshold": self.config.max_sector_exposure_pct,
                    "message": f"Sector exposure cap breached: '{top_sec}' is {max_sec:.2%} (max {self.config.max_sector_exposure_pct:.2%}).",
                })

        if var_95 is not None:
            checks["var_95_limit"] = var_95 <= self.config.max_var_95_pct
            if not checks["var_95_limit"]:
                warnings.append({
                    "severity": "CRITICAL" if var_95 > self.config.max_var_95_pct * 1.5 else "WARNING",
                    "category": "VAR_LIMIT",
                    "metric": "max_var_95_pct",
                    "value": round(var_95, 4),
                    "threshold": self.config.max_var_95_pct,
                    "message": f"Historical 95% VaR breached: {var_95:.2%} > limit {self.config.max_var_95_pct:.2%}.",
                })

        if cvar_95 is not None:
            checks["cvar_95_limit"] = cvar_95 <= self.config.max_cvar_95_pct
            if not checks["cvar_95_limit"]:
                warnings.append({
                    "severity": "CRITICAL",
                    "category": "CVAR_LIMIT",
                    "metric": "max_cvar_95_pct",
                    "value": round(cvar_95, 4),
                    "threshold": self.config.max_cvar_95_pct,
                    "message": f"Historical 95% Expected Shortfall (CVaR) breached: {cvar_95:.2%} > limit {self.config.max_cvar_95_pct:.2%}.",
                })

        if days_to_liquidate is not None:
            checks["liquidity_limit"] = days_to_liquidate <= self.config.max_days_to_liquidate
            if not checks["liquidity_limit"]:
                warnings.append({
                    "severity": "WARNING",
                    "category": "LIQUIDITY_LIMIT",
                    "metric": "max_days_to_liquidate",
                    "value": round(days_to_liquidate, 2),
                    "threshold": self.config.max_days_to_liquidate,
                    "message": f"Days to liquidate breached: {days_to_liquidate:.1f} days > max {self.config.max_days_to_liquidate:.1f} days.",
                })

        all_passed = all(checks.values())
        return all_passed, checks, warnings

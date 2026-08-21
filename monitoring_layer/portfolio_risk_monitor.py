"""
monitoring_layer/portfolio_risk_monitor.py
──────────────────────────────────────────
Institutional Portfolio Risk Monitor.

Monitors:
  - Value at Risk (VaR) — Parametric and Historical
  - Conditional VaR (CVaR / Expected Shortfall)
  - Concentration Risk (HHI, Gini, top-N weight)
  - Leverage
  - Gross / Net Exposure
  - Sector / Factor Tilt Alerts
  - Turnover Rate

Architecture:
  PortfolioRiskMonitor ──► AlertEngine
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from monitoring_layer.config import MonitoringConfig
from monitoring_layer.alert_engine import AlertEngine, AlertSeverity

logger = logging.getLogger(__name__)


class PortfolioRiskMonitor:
    """
    Real-time portfolio risk surveillance.

    Provides VaR/CVaR estimation, concentration, leverage, and turnover
    checks with automatic breach alerting.

    Usage:
        monitor = PortfolioRiskMonitor()
        weights = pd.Series({"AAPL": 0.10, "MSFT": 0.08, ...})
        returns_df = pd.DataFrame(...)   # rows=dates, cols=tickers
        report = monitor.check(weights, returns_df)
    """

    # Thresholds
    VAR_WARN_PCT: float = 0.02       # 2% daily VaR warning
    VAR_CRIT_PCT: float = 0.03       # 3% daily VaR critical
    HHI_WARN: float = 0.15           # Herfindahl-Hirschman Index (HHI) warning
    HHI_CRIT: float = 0.25           # HHI critical (very concentrated)
    LEVERAGE_WARN: float = 1.20      # 120% gross exposure warning
    LEVERAGE_CRIT: float = 1.50      # 150% gross exposure critical
    TURNOVER_WARN: float = 0.30      # 30% single-day turnover warning

    def __init__(
        self,
        config: Optional[MonitoringConfig] = None,
        alert_engine: Optional[AlertEngine] = None,
        confidence_level: float = 0.99,
    ):
        self.config = config or MonitoringConfig()
        self.alert_engine = alert_engine or AlertEngine(config=self.config)
        self.confidence_level = confidence_level

    # ── Main Check ────────────────────────────────────────────────────────────

    def check(
        self,
        weights: pd.Series,
        returns_df: pd.DataFrame,
        prev_weights: Optional[pd.Series] = None,
        sector_map: Optional[Dict[str, str]] = None,
        lookback_days: int = 252,
    ) -> Dict[str, Any]:
        """
        Full portfolio risk check.

        Args:
            weights:        Current portfolio weights (can be long-only or L/S).
            returns_df:     DataFrame of historical daily returns (rows=date, cols=ticker).
            prev_weights:   Previous period weights (for turnover calculation).
            sector_map:     Dict mapping ticker → sector (for concentration by sector).
            lookback_days:  Historical lookback for VaR/CVaR estimation.

        Returns:
            Portfolio risk report dict.
        """
        report: Dict[str, Any] = {
            "n_positions": int((weights.abs() > 1e-6).sum()),
            "gross_exposure": round(float(weights.abs().sum()), 4),
            "net_exposure": round(float(weights.sum()), 4),
        }

        # ── VaR / CVaR ───────────────────────────────────────────────────────
        var_report = self._compute_var_cvar(weights, returns_df, lookback_days)
        report.update(var_report)

        # ── Concentration ────────────────────────────────────────────────────
        conc = self._compute_concentration(weights)
        report.update(conc)

        # ── Leverage ─────────────────────────────────────────────────────────
        gross_exp = report["gross_exposure"]
        leverage_ok = True
        if gross_exp >= self.LEVERAGE_CRIT:
            self.alert_engine.fire(
                AlertSeverity.CRITICAL, "PORTFOLIO_RISK", "leverage",
                value=gross_exp, threshold=self.LEVERAGE_CRIT,
                message=f"CRITICAL leverage: gross exposure = {gross_exp:.1%}.",
            )
            leverage_ok = False
        elif gross_exp >= self.LEVERAGE_WARN:
            self.alert_engine.fire(
                AlertSeverity.WARNING, "PORTFOLIO_RISK", "leverage",
                value=gross_exp, threshold=self.LEVERAGE_WARN,
                message=f"High leverage: gross exposure = {gross_exp:.1%}.",
            )
            leverage_ok = False
        report["leverage_breach"] = not leverage_ok

        # ── Turnover ─────────────────────────────────────────────────────────
        if prev_weights is not None:
            turnover = self._compute_turnover(weights, prev_weights)
            report["turnover"] = round(turnover, 4)
            if turnover >= self.TURNOVER_WARN:
                self.alert_engine.fire(
                    AlertSeverity.WARNING, "PORTFOLIO_RISK", "turnover",
                    value=turnover, threshold=self.TURNOVER_WARN,
                    message=f"High portfolio turnover: {turnover:.1%}.",
                )
        else:
            report["turnover"] = None

        # ── Sector Concentration ──────────────────────────────────────────────
        if sector_map:
            sector_weights = self._compute_sector_weights(weights, sector_map)
            report["sector_weights"] = {k: round(v, 4) for k, v in sector_weights.items()}
            max_sector = max(sector_weights.values(), default=0.0)
            if max_sector > 0.35:
                self.alert_engine.fire(
                    AlertSeverity.WARNING, "PORTFOLIO_RISK", "sector_concentration",
                    value=max_sector, threshold=0.35,
                    message=f"Sector concentration {max_sector:.1%} exceeds 35%.",
                )

        return report

    # ── VaR / CVaR ────────────────────────────────────────────────────────────

    def _compute_var_cvar(
        self,
        weights: pd.Series,
        returns_df: pd.DataFrame,
        lookback_days: int,
    ) -> Dict[str, Any]:
        """Parametric and Historical VaR/CVaR."""
        shared = weights.index.intersection(returns_df.columns)
        if len(shared) < 3 or len(returns_df) < 30:
            return {
                "parametric_var": None,
                "historical_var": None,
                "cvar": None,
                "var_breach": False,
            }

        w = weights[shared].values
        R = returns_df[shared].iloc[-lookback_days:].dropna(how="all")
        port_returns = (R * w).sum(axis=1).dropna()

        if len(port_returns) < 20:
            return {
                "parametric_var": None,
                "historical_var": None,
                "cvar": None,
                "var_breach": False,
            }

        # Parametric VaR
        mu = float(port_returns.mean())
        sigma = float(port_returns.std())
        from scipy.stats import norm
        z = norm.ppf(1 - self.confidence_level)
        parametric_var = float(-(mu + z * sigma))  # Positive = loss

        # Historical VaR (99th percentile of losses)
        historical_var = float(-np.percentile(port_returns, (1 - self.confidence_level) * 100))

        # CVaR (Expected Shortfall): mean of returns below VaR threshold
        threshold = np.percentile(port_returns, (1 - self.confidence_level) * 100)
        cvar = float(-port_returns[port_returns <= threshold].mean()) if (port_returns <= threshold).any() else 0.0

        var_breach = False
        if historical_var >= self.VAR_CRIT_PCT:
            self.alert_engine.fire(
                AlertSeverity.CRITICAL, "PORTFOLIO_RISK", "var",
                value=historical_var, threshold=self.VAR_CRIT_PCT,
                message=f"CRITICAL VaR: {historical_var:.2%} exceeds {self.VAR_CRIT_PCT:.2%}.",
            )
            var_breach = True
        elif historical_var >= self.VAR_WARN_PCT:
            self.alert_engine.fire(
                AlertSeverity.WARNING, "PORTFOLIO_RISK", "var",
                value=historical_var, threshold=self.VAR_WARN_PCT,
                message=f"Elevated VaR: {historical_var:.2%} exceeds {self.VAR_WARN_PCT:.2%}.",
            )
            var_breach = True

        return {
            "parametric_var": round(parametric_var, 4),
            "historical_var": round(historical_var, 4),
            "cvar": round(cvar, 4),
            "var_confidence": self.confidence_level,
            "var_breach": var_breach,
        }

    # ── Concentration ─────────────────────────────────────────────────────────

    def _compute_concentration(self, weights: pd.Series) -> Dict[str, Any]:
        """Herfindahl-Hirschman Index and top-N concentration."""
        abs_w = weights.abs()
        total = abs_w.sum()
        if total < 1e-9:
            return {"hhi": 0.0, "top5_concentration": 0.0, "concentration_breach": False}

        w_norm = abs_w / total
        hhi = float((w_norm ** 2).sum())
        top5 = float(w_norm.nlargest(5).sum())

        breach = False
        if hhi >= self.HHI_CRIT:
            self.alert_engine.fire(
                AlertSeverity.CRITICAL, "PORTFOLIO_RISK", "concentration_hhi",
                value=hhi, threshold=self.HHI_CRIT,
                message=f"CRITICAL concentration HHI={hhi:.3f} (threshold: {self.HHI_CRIT}).",
            )
            breach = True
        elif hhi >= self.HHI_WARN:
            self.alert_engine.fire(
                AlertSeverity.WARNING, "PORTFOLIO_RISK", "concentration_hhi",
                value=hhi, threshold=self.HHI_WARN,
                message=f"Elevated concentration HHI={hhi:.3f} (threshold: {self.HHI_WARN}).",
            )
            breach = True

        return {
            "hhi": round(hhi, 4),
            "top5_concentration": round(top5, 4),
            "concentration_breach": breach,
        }

    # ── Turnover ──────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_turnover(weights: pd.Series, prev_weights: pd.Series) -> float:
        """One-way portfolio turnover as fraction of AUM."""
        all_tickers = weights.index.union(prev_weights.index)
        w_cur = weights.reindex(all_tickers, fill_value=0.0)
        w_prev = prev_weights.reindex(all_tickers, fill_value=0.0)
        return float((w_cur - w_prev).abs().sum() / 2.0)

    # ── Sector Weights ────────────────────────────────────────────────────────

    @staticmethod
    def _compute_sector_weights(
        weights: pd.Series, sector_map: Dict[str, str]
    ) -> Dict[str, float]:
        """Aggregate absolute weights by sector."""
        sector_weights: Dict[str, float] = {}
        total = weights.abs().sum()
        if total < 1e-9:
            return sector_weights
        for ticker, sector in sector_map.items():
            if ticker in weights.index:
                w = abs(float(weights[ticker])) / total
                sector_weights[sector] = sector_weights.get(sector, 0.0) + w
        return sector_weights

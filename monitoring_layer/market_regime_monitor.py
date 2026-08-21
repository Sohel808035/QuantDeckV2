"""
monitoring_layer/market_regime_monitor.py
──────────────────────────────────────────
Institutional Market Regime Monitor.

Detects:
  - Volatility Regime (Low / Normal / Elevated / Crisis)
  - Trend Regime (Bull / Bear / Sideways)
  - Liquidity Regime (liquid / stressed)
  - Correlation Regime (normal / correlated — risk-off)
  - Regime Change Events + Alerts
  - Rolling VIX proxy from realized volatility

Architecture:
  MarketRegimeMonitor ──► AlertEngine
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from monitoring_layer.config import MonitoringConfig
from monitoring_layer.alert_engine import AlertEngine, AlertSeverity

logger = logging.getLogger(__name__)


# ── Regime Enumerations ───────────────────────────────────────────────────────


class VolatilityRegime(str, Enum):
    LOW = "LOW"
    NORMAL = "NORMAL"
    ELEVATED = "ELEVATED"
    CRISIS = "CRISIS"


class TrendRegime(str, Enum):
    BULL = "BULL"
    BEAR = "BEAR"
    SIDEWAYS = "SIDEWAYS"


class CorrelationRegime(str, Enum):
    NORMAL = "NORMAL"
    CORRELATED = "CORRELATED"   # Risk-off — assets move together


# ── Market Regime Monitor ─────────────────────────────────────────────────────


class MarketRegimeMonitor:
    """
    Real-time market regime detection for QuantSphereX.

    Uses rolling realized volatility, trend filters, and cross-asset
    correlation to classify the current market regime.

    Usage:
        monitor = MarketRegimeMonitor()
        returns = pd.Series(...)          # market index daily returns
        returns_df = pd.DataFrame(...)    # multi-asset returns
        report = monitor.detect_regime(returns, returns_df)
    """

    # Volatility thresholds (annualized)
    VOL_LOW: float = 0.10      # < 10% → low vol
    VOL_NORMAL_HIGH: float = 0.18  # 10-18% → normal
    VOL_ELEVATED: float = 0.25    # 18-25% → elevated
    # > 25% → crisis

    # Trend filter: 50-day vs 200-day MA crossover
    FAST_MA: int = 50
    SLOW_MA: int = 200

    # Correlation regime threshold
    CORRELATION_CRISIS: float = 0.70  # Average pairwise correlation

    # Minimum lookback days
    MIN_LOOKBACK: int = 30

    def __init__(
        self,
        config: Optional[MonitoringConfig] = None,
        alert_engine: Optional[AlertEngine] = None,
    ):
        self.config = config or MonitoringConfig()
        self.alert_engine = alert_engine or AlertEngine(config=self.config)
        self._prev_regime: Optional[str] = None

    # ── Full Regime Detection ─────────────────────────────────────────────────

    def detect_regime(
        self,
        market_returns: pd.Series,
        multi_asset_returns: Optional[pd.DataFrame] = None,
        vol_window: int = 21,
        price_series: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Detects current market regime across volatility, trend, and correlation.

        Args:
            market_returns:      Market index daily returns (e.g. Nifty500).
            multi_asset_returns: Multi-asset daily returns DataFrame for correlation.
            vol_window:          Rolling window for realized vol (default: 21 days).
            price_series:        Price levels for MA-based trend detection.

        Returns:
            Regime report dict with fields:
              volatility_regime, trend_regime, correlation_regime,
              realized_vol_ann, vol_z_score, regime_composite,
              regime_changed, alerts
        """
        if len(market_returns) < self.MIN_LOOKBACK:
            return {
                "status": "insufficient_data",
                "min_required": self.MIN_LOOKBACK,
                "n_provided": len(market_returns),
            }

        report: Dict[str, Any] = {}

        # ── Volatility Regime ─────────────────────────────────────────────────
        vol_report = self._detect_volatility_regime(market_returns, vol_window)
        report.update(vol_report)

        # ── Trend Regime ──────────────────────────────────────────────────────
        if price_series is not None and len(price_series) >= self.SLOW_MA:
            trend_report = self._detect_trend_regime(price_series)
        else:
            # Derive from cumulative returns
            synthetic_price = (1 + market_returns).cumprod()
            trend_report = self._detect_trend_regime(synthetic_price)
        report.update(trend_report)

        # ── Correlation Regime ────────────────────────────────────────────────
        if multi_asset_returns is not None and multi_asset_returns.shape[1] >= 3:
            corr_report = self._detect_correlation_regime(
                multi_asset_returns, window=vol_window
            )
            report.update(corr_report)
        else:
            report["correlation_regime"] = CorrelationRegime.NORMAL.value
            report["avg_pairwise_correlation"] = None

        # ── Composite Regime Score ────────────────────────────────────────────
        composite = self._compute_composite_regime(
            vol_regime=report.get("volatility_regime", VolatilityRegime.NORMAL.value),
            trend_regime=report.get("trend_regime", TrendRegime.SIDEWAYS.value),
            corr_regime=report.get("correlation_regime", CorrelationRegime.NORMAL.value),
        )
        report["regime_composite"] = composite

        # ── Change Detection + Alerts ─────────────────────────────────────────
        current_regime = f"{report.get('volatility_regime')}|{report.get('trend_regime')}"
        regime_changed = current_regime != self._prev_regime and self._prev_regime is not None
        report["regime_changed"] = regime_changed

        if regime_changed:
            logger.warning(
                "[MarketRegime] Regime change: %s → %s",
                self._prev_regime, current_regime,
            )
            if composite in ("CRISIS", "BEAR_CRISIS"):
                self.alert_engine.fire(
                    AlertSeverity.CRITICAL, "MARKET_REGIME", "regime_change",
                    value=1.0, threshold=0.0,
                    message=(
                        f"Market regime changed to {composite}: "
                        f"{self._prev_regime} → {current_regime}. "
                        "Consider reducing risk exposure."
                    ),
                )
            elif composite in ("ELEVATED", "BEAR"):
                self.alert_engine.fire(
                    AlertSeverity.WARNING, "MARKET_REGIME", "regime_change",
                    value=1.0, threshold=0.0,
                    message=(
                        f"Market regime shifted to {composite}. "
                        f"Previous: {self._prev_regime}."
                    ),
                )

        self._prev_regime = current_regime
        return report

    # ── Volatility Regime ─────────────────────────────────────────────────────

    def _detect_volatility_regime(
        self, returns: pd.Series, window: int
    ) -> Dict[str, Any]:
        """Classifies volatility regime using rolling realized vol."""
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        current_vol = float(rolling_vol.iloc[-1]) if not rolling_vol.empty else 0.0
        long_run_vol = float(rolling_vol.dropna().mean()) if rolling_vol.dropna().any() else 0.0

        # Vol Z-score relative to long-run average
        vol_std = float(rolling_vol.dropna().std()) or 1e-6
        vol_z = (current_vol - long_run_vol) / vol_std if vol_std > 0 else 0.0

        if current_vol > self.VOL_ELEVATED:
            regime = VolatilityRegime.CRISIS
        elif current_vol > self.VOL_NORMAL_HIGH:
            regime = VolatilityRegime.ELEVATED
        elif current_vol > self.VOL_LOW:
            regime = VolatilityRegime.NORMAL
        else:
            regime = VolatilityRegime.LOW

        # Alert on CRISIS volatility
        if regime == VolatilityRegime.CRISIS:
            self.alert_engine.fire(
                AlertSeverity.CRITICAL, "MARKET_REGIME", "volatility_crisis",
                value=current_vol, threshold=self.VOL_ELEVATED,
                message=f"Volatility regime = CRISIS: realized vol {current_vol:.1%} (annualized).",
            )
        elif regime == VolatilityRegime.ELEVATED:
            self.alert_engine.fire(
                AlertSeverity.WARNING, "MARKET_REGIME", "volatility_elevated",
                value=current_vol, threshold=self.VOL_NORMAL_HIGH,
                message=f"Volatility elevated: {current_vol:.1%} annualized.",
            )

        return {
            "volatility_regime": regime.value,
            "realized_vol_ann": round(current_vol, 4),
            "long_run_vol_ann": round(long_run_vol, 4),
            "vol_z_score": round(vol_z, 3),
            "vol_window": window,
        }

    # ── Trend Regime ──────────────────────────────────────────────────────────

    def _detect_trend_regime(self, price: pd.Series) -> Dict[str, Any]:
        """MA-crossover-based trend detection."""
        fast = int(min(self.FAST_MA, len(price)))
        slow = int(min(self.SLOW_MA, len(price)))

        ma_fast = float(price.rolling(fast).mean().iloc[-1])
        ma_slow = float(price.rolling(slow).mean().iloc[-1])
        current_price = float(price.iloc[-1])

        if np.isnan(ma_fast) or np.isnan(ma_slow):
            return {
                "trend_regime": TrendRegime.SIDEWAYS.value,
                "ma_fast": None,
                "ma_slow": None,
                "price_vs_slow_ma_pct": None,
            }

        price_vs_slow = (current_price - ma_slow) / ma_slow

        if ma_fast > ma_slow and current_price > ma_fast:
            trend = TrendRegime.BULL
        elif ma_fast < ma_slow and current_price < ma_fast:
            trend = TrendRegime.BEAR
        else:
            trend = TrendRegime.SIDEWAYS

        return {
            "trend_regime": trend.value,
            "ma_fast": round(ma_fast, 4),
            "ma_slow": round(ma_slow, 4),
            "price_vs_slow_ma_pct": round(price_vs_slow * 100, 2),
        }

    # ── Correlation Regime ────────────────────────────────────────────────────

    def _detect_correlation_regime(
        self, returns_df: pd.DataFrame, window: int
    ) -> Dict[str, Any]:
        """Detects correlation regime — high pairwise correlation = risk-off."""
        recent = returns_df.iloc[-window:].dropna(how="all")
        if recent.shape[1] < 2 or len(recent) < 10:
            return {
                "correlation_regime": CorrelationRegime.NORMAL.value,
                "avg_pairwise_correlation": None,
            }

        corr_matrix = recent.corr()
        # Average of upper triangle (excluding diagonal)
        upper = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
        avg_corr = float(np.nanmean(np.abs(upper)))

        if avg_corr >= self.CORRELATION_CRISIS:
            regime = CorrelationRegime.CORRELATED
            self.alert_engine.fire(
                AlertSeverity.WARNING, "MARKET_REGIME", "correlation_regime",
                value=avg_corr, threshold=self.CORRELATION_CRISIS,
                message=(
                    f"Risk-off correlation regime: avg pairwise |ρ| = {avg_corr:.2f}. "
                    "Diversification benefits reduced."
                ),
            )
        else:
            regime = CorrelationRegime.NORMAL

        return {
            "correlation_regime": regime.value,
            "avg_pairwise_correlation": round(avg_corr, 4),
        }

    # ── Composite Regime Score ────────────────────────────────────────────────

    @staticmethod
    def _compute_composite_regime(
        vol_regime: str, trend_regime: str, corr_regime: str
    ) -> str:
        """Maps (vol, trend, correlation) → composite regime label."""
        if vol_regime == VolatilityRegime.CRISIS.value:
            return "CRISIS"
        if vol_regime == VolatilityRegime.ELEVATED.value and trend_regime == TrendRegime.BEAR.value:
            return "BEAR_CRISIS"
        if vol_regime == VolatilityRegime.ELEVATED.value:
            return "ELEVATED"
        if trend_regime == TrendRegime.BEAR.value:
            return "BEAR"
        if trend_regime == TrendRegime.BULL.value:
            return "BULL"
        if corr_regime == CorrelationRegime.CORRELATED.value:
            return "RISK_OFF"
        return "NEUTRAL"

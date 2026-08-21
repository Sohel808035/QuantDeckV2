"""
portfolio_layer/comparison.py
───────────────────────────────
Phase 8: Multi-Optimizer Performance Comparison Suite.
Evaluates and benchmarks all 7 portfolio optimization strategies:
  1. Equal Weight (Baseline)
  2. Hierarchical Risk Parity (HRP)
  3. Risk Parity (ERC)
  4. Minimum Variance
  5. Confidence-Weighted Allocation
  6. Bounded Kelly Criterion
  7. Volatility Targeting

Metrics evaluated:
  - CAGR
  - Sharpe Ratio
  - Sortino Ratio
  - Maximum Drawdown
  - Information Ratio
  - Turnover
  - Volatility
  - Annual Return
  - Transaction Cost Impact
"""

from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from portfolio_layer.optimizer import PortfolioOptimizer
from execution_layer.backtester import Backtester

logger = logging.getLogger(__name__)


class PortfolioComparisonSuite:
    """Runs parallel optimization backtests across all strategies and computes comparison metrics."""

    OPTIMIZERS = [
        "equal_weight",
        "hrp",
        "risk_parity",
        "min_variance",
        "confidence_weighted",
        "kelly",
        "volatility_targeting",
    ]

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        transaction_cost: float = 0.0015,
        target_volatility: float = 0.14,
    ):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.target_volatility = target_volatility
        self.optimizer = PortfolioOptimizer()

    def calculate_metrics(
        self,
        equity_curve: pd.Series,
        daily_returns: pd.Series,
        weights_df: pd.DataFrame,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """Calculates institutional metrics for a single optimization strategy."""
        if equity_curve.empty or len(daily_returns) < 2:
            return {}

        n_days = len(daily_returns)
        n_years = max(n_days / 252.0, 1e-4)

        final_eq = float(equity_curve.iloc[-1])
        start_eq = float(equity_curve.iloc[0])

        cagr = (final_eq / start_eq) ** (1.0 / n_years) - 1.0
        ann_return = float(daily_returns.mean() * 252.0)
        ann_vol = float(daily_returns.std() * np.sqrt(252.0))

        # Sharpe ratio
        rf = 0.05
        excess_returns = daily_returns - (rf / 252.0)
        sharpe = float(excess_returns.mean() / (daily_returns.std() + 1e-8) * np.sqrt(252.0))

        # Sortino ratio
        downside_std = float(daily_returns[daily_returns < 0].std() * np.sqrt(252.0))
        sortino = float((ann_return - rf) / (downside_std + 1e-8))

        # Maximum Drawdown
        cum_max = equity_curve.cummax()
        drawdown = (equity_curve - cum_max) / cum_max
        max_dd = float(drawdown.min())

        # Turnover calculation
        trade_diff = weights_df.diff().abs().sum(axis=1) / 2.0
        ann_turnover = float(trade_diff.sum() / max(n_years, 1.0))

        # Transaction cost impact (bps per year)
        tx_cost_drag_bp = ann_turnover * (self.transaction_cost * 10000.0)

        # Information Ratio against benchmark
        info_ratio = 0.0
        if benchmark_returns is not None and not benchmark_returns.empty:
            common = daily_returns.index.intersection(benchmark_returns.index)
            if len(common) > 10:
                active_ret = daily_returns.loc[common] - benchmark_returns.loc[common]
                tracking_error = float(active_ret.std() * np.sqrt(252.0))
                info_ratio = float((active_ret.mean() * 252.0) / (tracking_error + 1e-8))

        return {
            "cagr": round(cagr, 4),
            "annual_return": round(ann_return, 4),
            "volatility": round(ann_vol, 4),
            "sharpe_ratio": round(sharpe, 4),
            "sortino_ratio": round(sortino, 4),
            "max_drawdown": round(max_dd, 4),
            "information_ratio": round(info_ratio, 4),
            "turnover": round(ann_turnover, 4),
            "transaction_cost_impact_bps": round(tx_cost_drag_bp, 2),
        }

    def run_comparison(
        self,
        scores_df: pd.DataFrame,
        stock_returns: pd.DataFrame,
        confidence_df: Optional[pd.DataFrame] = None,
        regime_exposure: Optional[pd.Series] = None,
        adv_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Runs parallel backtest comparison across all 7 portfolio optimization strategies."""
        logger.info("=" * 70)
        logger.info("PHASE 8 — MULTI-OPTIMIZER PERFORMANCE COMPARISON SUITE")
        logger.info("=" * 70)

        results = []
        benchmark_returns = stock_returns.mean(axis=1).fillna(0.0)

        for opt_name in self.OPTIMIZERS:
            try:
                # Generate weights schedule for opt_name
                weight_schedule = {}
                for dt in scores_df.index:
                    daily_scores = scores_df.loc[dt].dropna()
                    active_tickers = set(daily_scores.head(45).index)

                    if not active_tickers:
                        continue

                    # Slice returns window for covariance calculation
                    dt_idx = stock_returns.index.get_indexer([dt], method="pad")[0]
                    ret_window = stock_returns.iloc[max(0, dt_idx - 126): dt_idx + 1]

                    adv_dt = None
                    if adv_data is not None:
                        if isinstance(adv_data, pd.DataFrame) and dt in adv_data.index:
                            adv_dt = adv_data.loc[dt]
                        elif isinstance(adv_data, pd.Series):
                            adv_dt = adv_data

                    weights = self.optimizer.optimize(
                        selected_tickers=active_tickers,
                        optimizer_name=opt_name,
                        returns_df=ret_window,
                        alpha_scores=daily_scores,
                        adv_data=adv_dt,
                        confidence_df=confidence_df,
                        target_volatility=self.target_volatility,
                    )
                    weight_schedule[dt] = weights

                if not weight_schedule:
                    continue

                all_d = sorted(weight_schedule.keys())
                all_t = sorted(set().union(*[w.index for w in weight_schedule.values()]))
                weights_df = pd.DataFrame(index=all_d, columns=all_t, dtype=float).fillna(0.0)
                for d in all_d:
                    weights_df.loc[d, weight_schedule[d].index] = weight_schedule[d].values

                # Run backtest
                bt = Backtester(
                    initial_capital=self.initial_capital,
                    transaction_cost=self.transaction_cost,
                    target_vol=self.target_volatility,
                    apply_vol_targeting=True,
                )
                bt_res = bt.run_backtest(weights_df, stock_returns, regime_exposure, adv_data=adv_data)

                metrics = self.calculate_metrics(
                    equity_curve=bt_res["equity_curve"],
                    daily_returns=bt_res["daily_returns"],
                    weights_df=weights_df,
                    benchmark_returns=benchmark_returns,
                )
                metrics["strategy"] = opt_name.upper()
                results.append(metrics)

            except Exception as exc:
                import traceback
                logger.warning(f"[ComparisonSuite] Error evaluating '{opt_name}': {exc}\n{traceback.format_exc()}")

        if not results:
            return pd.DataFrame()

        df_res = pd.DataFrame(results).set_index("strategy")
        return df_res

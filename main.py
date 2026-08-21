"""
QuantDeck — CQRO Institutional Alpha Engine
════════════════════════════════════════════════════════════════════════════════
Chief Quantitative Research Officer Mandate Implementation.

Execution Order:
  §I   Data Integrity & Leakage Control
  §II  Alpha Engine Design (Orthogonal & Regime-Aware)
  §III Model Training Protocol (Ensemble XGBoost, Walk-Forward, Overfitting Guard)
  §IV  Pure Alpha Validation (IC, t-stat, Decile Spread)
  §V   Portfolio Defensive Construction
  §VI  Transaction Cost & Execution Realism
  §VII Regime Robustness Testing
  §VIII Stress Testing
  §IX  Final Institutional Decision Matrix
  §X   Institutional Monitoring Platform

Absolute Rule:
  Alpha quality precedes engineering.
  Capital preservation > headline Sharpe.
  Robustness > peak return.
"""

from __future__ import annotations
import logging
import warnings
import sys
import os

# Force UTF-8 output on Windows terminals
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
# Suppress yfinance error noise
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)

# ── Local imports ─────────────────────────────────────────────────────────────
from data_layer.universe   import get_universe, UniverseManager
from data_layer.ingestor   import YFinanceIngestor, MacroDataIngestor
from data_layer.storage    import ParquetCache
from feature_layer.implementations import (
    compute_stock_features,
    post_process_features,
    apply_cross_sectional_rank,
    apply_sector_neutralization,
    drop_highly_correlated_features,
    FEATURE_COLS,
)
from alpha_layer.targets              import build_target_panel, TARGET_COL
from feature_store                    import FeatureStoreWriter, FeatureStoreReader, FeatureRegistry
from alpha_layer.xgboost_trainer      import EnsembleAlphaModel
from ml_layer.explainability          import InstitutionalExplainabilityEngine
from ml_layer.confidence              import full_confidence_report
from ml_layer.experiment_tracker      import ExperimentTracker
from alpha_layer.governance           import ModelRegistry, ModelMetadata, save_model, load_model, compare_models, verify_reproducibility
from alpha_layer.walk_forward         import WalkForwardEngine
from alpha_layer.pure_alpha_validator import evaluate_pure_alpha
from portfolio_layer.ranking     import CrossSectionalRanker
from portfolio_layer.optimizer   import PortfolioOptimizer
from portfolio_layer.comparison  import PortfolioComparisonSuite
from risk_layer.engine          import InstitutionalRiskEngine
from risk_layer.regime_model     import compute_regime_exposure
from risk_layer.vol_targeting    import compute_vol_target_scalar
from risk_layer.regime_robustness import run_regime_robustness
from execution_layer.backtester  import Backtester
from execution_layer.stress_tester import run_stress_tests
from monitoring_layer             import MonitoringLayer

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
START_DATE          = "2005-01-01"
TRAIN_WINDOW_YEARS  = 3        
REBALANCE_MONTHS    = 1        # Higher frequency for alpha capture
REBALANCE_HORIZON   = 60       # Rank target horizon
TOP_N               = 45       # Baseline (dynamic override in Step 3)
BUFFER_N            = 65       # Baseline (dynamic override in Step 3)
INITIAL_CAPITAL     = 100_000.0
TARGET_VOL          = 0.14
TRANSACTION_COST    = 0.0015   
IMPACT_COEFF        = 0.1
IC_EXPOSURE_THRESH  = 0.03     
VOL_EXPOSURE_THRESH = 25.0     
TURNOVER_PENALTY    = 0.015    # Step 3 mandate

pd.set_option("display.max_columns", None)


# ══════════════════════════════════════════════════════════════════════════════
# §I — DATA INTEGRITY
# ══════════════════════════════════════════════════════════════════════════════
def step1_fetch_data():
    logger.info("=" * 70)
    logger.info("§I — DATA FETCH & INTEGRITY CHECKS")
    logger.info("=" * 70)
    
    # Clear existing caches for fresh feature engine
    cache_dir = Path("data_cache")
    if cache_dir.exists():
        for f in cache_dir.glob("stock_*.parquet"):
            try: f.unlink()
            except Exception: pass
        for f in cache_dir.glob("walk_forward_cache.pkl"):
            try: f.unlink()
            except Exception: pass
        logger.info("  [Cache] Cleared stock and walk-forward caches for fresh run.")

    cache  = ParquetCache()
    ingest = YFinanceIngestor(cache=cache)
    macro  = MacroDataIngestor(cache=cache)

    tickers = get_universe()
    logger.info(f"Universe: {len(tickers)} tickers")

    stock_panel = ingest.fetch_daily_data(tickers, start_date=START_DATE)
    nifty_df    = macro.fetch_nifty50(start_date=START_DATE)
    vix_df      = macro.fetch_india_vix(start_date=START_DATE)

    # §I.6 — Log missing data events
    n_tickers    = stock_panel.index.get_level_values("Ticker").nunique()
    n_days_total = stock_panel.index.get_level_values("Date").nunique()
    missing_ct   = stock_panel["Close"].isna().sum()
    logger.info(f"Stock panel: {len(stock_panel):,} rows | {n_tickers} tickers | {n_days_total} days")
    if missing_ct > 0:
        logger.warning(f"  §I.6 Missing data: {missing_ct:,} NaN Close prices detected.")

    # §I.3 — Verify index alignment
    stock_dates = stock_panel.index.get_level_values("Date").unique()
    nifty_dates = nifty_df.index
    overlap = len(stock_dates.intersection(nifty_dates))
    if overlap < 100:
        raise ValueError(f"§I.3 ALIGNMENT MISMATCH: Only {overlap} common dates during fetch.")
    
    return stock_panel, nifty_df, vix_df


# ══════════════════════════════════════════════════════════════════════════════
# §II — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
def step2_build_features(stock_panel: pd.DataFrame) -> pd.DataFrame:
    logger.info("=" * 70)
    logger.info("§II — ALPHA ENGINE DESIGN (Step 1-2 Upgrade)")
    logger.info("=" * 70)

    tickers  = stock_panel.index.get_level_values(1).unique()
    univ_mgr = UniverseManager()
    sector_map = univ_mgr.get_sector_mapping()

    daily_rets = stock_panel["Close"].unstack().pct_change().fillna(0)
    sector_rets = pd.DataFrame(index=daily_rets.index)
    for sec in sorted(set(sector_map.values())):
        sec_t = [t for t, s in sector_map.items() if s == sec and t in daily_rets.columns]
        if sec_t:
            sector_rets[sec] = daily_rets[sec_t].mean(axis=1)

    logger.info(f"  Processing {len(tickers)} tickers with upgraded library factors...")
    feature_frames = []
    for ticker in tickers:
        try:
            tkr_df = stock_panel.xs(ticker, level="Ticker")
            if len(tkr_df.dropna(subset=["Close"])) < 252:
                continue

            sec         = sector_map.get(ticker, "Other")
            context_ret = sector_rets[sec] if sec in sector_rets.columns else daily_rets.mean(axis=1)
            
            feat = compute_stock_features(tkr_df, context_ret=context_ret)
            feat["Ticker"] = ticker
            feat = feat.set_index("Ticker", append=True)
            
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                if col in tkr_df.columns:
                    feat[col] = tkr_df[col].values
            feature_frames.append(feat)
        except Exception as exc:
            logger.warning(f"  Skipping {ticker}: {exc}")

    panel = pd.concat(feature_frames).sort_index()
    
    panel = post_process_features(panel)
    panel = apply_sector_neutralization(panel, sector_map)
    panel = apply_cross_sectional_rank(panel)
    panel = drop_highly_correlated_features(panel, threshold=0.6)
    
    # Phase 3 Feature Store Persistence
    try:
        writer = FeatureStoreWriter()
        manifest = writer.write(
            panel=panel,
            input_dataset_name="stock_panel",
            input_dataset_version="v1.0",
            transform_name="FeatureFactoryPipeline"
        )
        logger.info(f"  [FeatureStore] Persisted feature panel version '{manifest.version_id}' ({manifest.row_count:,} rows)")
    except Exception as exc:
        logger.warning(f"  [FeatureStore] Persistence warning: {exc}")

    return panel


# ══════════════════════════════════════════════════════════════════════════════
# §III — WALK-FORWARD MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════
def step3_walk_forward(full_panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, EnsembleAlphaModel]:
    logger.info("=" * 70)
    logger.info("§III — WALK-FORWARD TRAINING (Ensemble XGBoost, Overfitting Guard)")
    logger.info("=" * 70)

    exp_tracker = ExperimentTracker()

    features_cols = sorted([c for c in full_panel.columns if c in FEATURE_COLS and full_panel[c].notna().any()])
    logger.info(f"  Institutional active features: {features_cols}")

    engine   = WalkForwardEngine(train_years=TRAIN_WINDOW_YEARS, rebalance_months=REBALANCE_MONTHS, horizon_days=60, embargo_days=10)
    ranker   = CrossSectionalRanker(top_n=TOP_N, buffer_n=BUFFER_N)
    opt      = PortfolioOptimizer()
    univ_mgr = UniverseManager()

    adv_pivot = (full_panel["Close"] * full_panel["Volume"]).unstack(level=1)

    weight_schedule: dict = {}
    scores_history:  dict = {}
    current_holdings: set = set()
    prev_weights = pd.Series(dtype=float)
    overfitting_count = 0
    total_windows     = 0
    latest_model      = None

    for i, (train_df, pred_date) in enumerate(engine.generate_splits(full_panel)):
        total_windows += 1
        X_train = train_df[features_cols].dropna()
        y_train = train_df.loc[X_train.index, TARGET_COL]

        if len(X_train) < 500:
            logger.warning(f"  Skipping {pred_date.date()}: insufficient training rows ({len(X_train)}).")
            continue

        model = EnsembleAlphaModel(n_models=2)
        fit_res = model.fit(X_train, y_train, val_split=0.2)
        latest_model = model

        if fit_res.get("overfit_score", 0) > 0.05:
            overfitting_count += 1

        # Phase 2: Experiment Tracking Logging
        try:
            run_id_str = exp_tracker.start_run(
                experiment_name="walk_forward_alpha",
                tags={"window": str(i), "date": str(pred_date.date())}
            )
            exp_tracker.log_metrics(
                run_id=run_id_str,
                experiment_name="walk_forward_alpha",
                metrics={
                    "val_ic": fit_res.get("val_ic", 0.0),
                    "train_ic": fit_res.get("train_ic", 0.0),
                    "overfit_score": fit_res.get("overfit_score", 0.0),
                },
            )
        except Exception as tracker_exc:
            logger.warning(f"  [ExperimentTracker] Logging warning: {tracker_exc}")

        # Model Governance Integration
        import hashlib
        ds_hash = f"{X_train.index.get_level_values(0).min().strftime('%Y%m%d')}_{X_train.index.get_level_values(0).max().strftime('%Y%m%d')}"
        feat_hash = hashlib.md5(",".join(features_cols).encode()).hexdigest()[:8]
        
        val_ic = fit_res.get("val_ic", 0.0)
        status = "Production" if val_ic >= 0.02 else "Experimental"
        
        metadata_args = {
            "model_id": "EnsembleAlphaModel",
            "version": f"1.{i}.0",
            "dataset_version": ds_hash,
            "feature_version": feat_hash,
            "hyperparameters": model.get_hyperparameters(),
            "validation_ic": float(val_ic),
            "train_ic": float(fit_res.get("train_ic", 0.0)),
            "sharpe": 0.0,
            "drawdown": 0.0,
            "status": status,
            "notes": f"Walk-forward training window {i} on date {pred_date.date()}"
        }
        
        registry_inst = ModelRegistry()
        saved_path = save_model(model, metadata_args, registry=registry_inst)
        logger.info(f"  [Governance] Saved model version 1.{i}.0 to {saved_path}")
        
        loaded_model, loaded_meta = load_model("EnsembleAlphaModel", f"1.{i}.0", registry=registry_inst)
        verify_reproducibility(model, loaded_model, X_train.head(100))

        # Signal Smoothing (5-day MA)
        all_dates = full_panel.index.get_level_values("Date").unique().sort_values()
        try:
            idx_rebal = int(all_dates.get_loc(pred_date))  # type: ignore[arg-type]
        except KeyError:
            continue
        smooth_window = all_dates[max(0, idx_rebal - 4): idx_rebal + 1]

        monthly_scores_list = []
        for d in smooth_window:
            try:
                cs_feat = full_panel.xs(d, level="Date")[features_cols].dropna()
                pit_univ = univ_mgr.get_universe(d)
                cs_feat  = cs_feat.reindex([t for t in cs_feat.index if t in pit_univ])
                if not cs_feat.empty:
                    monthly_scores_list.append(model.predict(cs_feat))
            except KeyError:
                continue

        if not monthly_scores_list:
            continue
        scores = pd.concat(monthly_scores_list, axis=1).mean(axis=1)

        dynamic_top_n = max(12, int(len(scores) * 0.15))
        dynamic_buffer = int(dynamic_top_n * 1.4)
        
        ranker.top_n = dynamic_top_n
        ranker.buffer_n = dynamic_buffer
        
        new_portfolio = ranker.select_portfolio(scores, current_holdings, pred_date)
        adv_current   = adv_pivot.loc[pred_date] if pred_date in adv_pivot.index else None
        weights       = opt.equal_weight(new_portfolio, adv_data=adv_current, portfolio_value=INITIAL_CAPITAL)
        
        univ_mgr = UniverseManager()
        weights  = opt.sector_neutralize(weights, univ_mgr.get_sector_mapping(), univ_mgr.get_benchmark_sector_weights())
        weights  = opt.apply_turnover_penalty(prev_weights, weights, threshold=TURNOVER_PENALTY)

        weight_schedule[pred_date] = weights
        scores_history[pred_date]  = scores
        current_holdings           = new_portfolio
        prev_weights               = weights

    if total_windows > 0:
        overfit_pct = overfitting_count / total_windows
        logger.info(f"\n  [Overfitting Audit] {overfitting_count}/{total_windows} windows triggered regularization ({overfit_pct:.1%})")

    all_d = sorted(weight_schedule.keys())
    all_t = sorted(set().union(*[w.index for w in weight_schedule.values()]))
    weight_df = pd.DataFrame(index=all_d, columns=all_t, dtype=float).fillna(0.0)
    scores_df = pd.DataFrame(index=all_d, columns=all_t, dtype=float).fillna(0.0)
    for date in all_d:
        weight_df.loc[date, weight_schedule[date].index] = weight_schedule[date].values
        scores_df.loc[date, scores_history[date].index]  = scores_history[date].values

    return weight_df, scores_df, latest_model  # type: ignore[return-value]


# ══════════════════════════════════════════════════════════════════════════════
# Phase 4 & 5 Integration: Explainability & Confidence Engine
# ══════════════════════════════════════════════════════════════════════════════
def step3b_explainability_and_confidence(model: EnsembleAlphaModel, full_panel: pd.DataFrame):
    logger.info("=" * 70)
    logger.info("§III.B — EXPLAINABILITY & CONFIDENCE ENGINES (Phases 4 & 5)")
    logger.info("=" * 70)
    
    if model is None:
        logger.warning("  No trained model available for explainability/confidence.")
        return

    features_cols = sorted([c for c in full_panel.columns if c in FEATURE_COLS and full_panel[c].notna().any()])
    recent_date = full_panel.index.get_level_values("Date").max()
    sample_df = full_panel.xs(recent_date, level="Date")[features_cols].dropna()

    if sample_df.empty:
        logger.warning("  Empty sample DataFrame for explainability.")
        return

    # Phase 4: Institutional Explainability Engine
    try:
        explainer = InstitutionalExplainabilityEngine(model)
        importance_df = explainer.get_global_feature_importance(sample_df, max_samples=500)
        out_exp = Path("reports/prediction_explanations.csv")
        out_exp.parent.mkdir(parents=True, exist_ok=True)
        if not importance_df.empty:
            importance_df.to_csv(out_exp, index=False)
            logger.info(f"  ✅ [Phase 4 Explainability] Global Feature Importance saved to {out_exp.resolve()}")
    except Exception as exc:
        logger.warning(f"  [Explainability] Warning: {exc}")

    # Phase 5: Confidence Engine
    try:
        conf_df = full_confidence_report(model, sample_df)
        out_conf = Path("reports/confidence_report.csv")
        conf_df.to_csv(out_conf)
        logger.info(f"  ✅ [Phase 5 Confidence] Prediction Confidence Report saved to {out_conf.resolve()}")
    except Exception as exc:
        logger.warning(f"  [Confidence] Warning: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# §VI — PRODUCTION BACKTEST
# ══════════════════════════════════════════════════════════════════════════════
def step4_production_backtest(
    weight_df:       pd.DataFrame,
    stock_panel:     pd.DataFrame,
    nifty_df:        pd.DataFrame,
    vix_df:          pd.DataFrame,
    ic_series:       pd.Series,
) -> Dict[str, Any]:
    logger.info("=" * 70)
    logger.info("§VI — PRODUCTION BACKTEST (Full Cost & Execution Realism)")
    logger.info("=" * 70)

    close_prices  = stock_panel["Close"].unstack(level="Ticker").sort_index()
    stock_returns = close_prices.pct_change().fillna(0.0)
    regime_exposure = compute_regime_exposure(nifty_df, vix_df=vix_df)
    adv_data = (stock_panel["Close"] * stock_panel["Volume"]).unstack(level=1).rolling(20, min_periods=5).mean().ffill()

    common_dates = weight_df.index.intersection(stock_returns.index)
    if len(common_dates) < 10:
        raise ValueError(f"§I.3 ALIGNMENT MISMATCH: Only {len(common_dates)} common dates in weight↔returns.")
    logger.info(f"  [I.3] Weight <-> Returns date overlap: {len(common_dates)} dates OK")

    bt = Backtester(
        initial_capital=INITIAL_CAPITAL,
        transaction_cost=TRANSACTION_COST,
        target_vol=TARGET_VOL,
        apply_vol_targeting=True,
    )
    results = bt.run_backtest(weight_df, stock_returns, regime_exposure, adv_data=adv_data, impact_coeff=IMPACT_COEFF)

    if results["ann_turnover"] > 1e-6 and results["ann_fixed_cost_bp"] < 1.0:
        raise ValueError("§VI COST BUG: Positive turnover detected but zero fixed cost reported.")

    logger.info(f"  CAGR             : {results['cagr']:.2%}")
    logger.info(f"  Net Sharpe       : {results['sharpe_ratio']:.2f}")
    logger.info(f"  Ann. Volatility  : {results['ann_vol']:.2%}")
    logger.info(f"  Max Drawdown     : {results['max_drawdown']:.2%}")
    logger.info(f"  Ann. Turnover    : {results['ann_turnover']:.1%}")
    logger.info(f"  Fixed Cost       : {results['ann_fixed_cost_bp']:.1f} bps/yr")
    logger.info(f"  Impact Cost      : {results['ann_impact_cost_bp']:.1f} bps/yr")
    logger.info(f"  Total Cost Drag  : {results['ann_fixed_cost_bp'] + results['ann_impact_cost_bp']:.1f} bps/yr")

    return results


def step4b_portfolio_comparison(
    scores_df: pd.DataFrame,
    stock_panel: pd.DataFrame,
    nifty_df: pd.DataFrame,
    vix_df: Optional[pd.DataFrame] = None,  # noqa: F821
) -> pd.DataFrame:
    """Phase 8 Portfolio Engine 2.0 Multi-Optimizer Performance Comparison."""
    logger.info("=" * 70)
    logger.info("§VI.B — PHASE 8 PORTFOLIO ENGINE 2.0 MULTI-OPTIMIZER COMPARISON")
    logger.info("=" * 70)

    close_prices = stock_panel["Close"].unstack(level="Ticker").sort_index()
    stock_returns = close_prices.pct_change().fillna(0.0)
    regime_exposure = compute_regime_exposure(nifty_df, vix_df=vix_df)
    adv_data = (stock_panel["Close"] * stock_panel["Volume"]).unstack(level=1).rolling(20, min_periods=5).mean().ffill()

    confidence_path = Path("reports/confidence_report.csv")
    confidence_df = None
    if confidence_path.exists():
        try:
            confidence_df = pd.read_csv(confidence_path).set_index("symbol")
        except Exception:
            pass

    comp_suite = PortfolioComparisonSuite(
        initial_capital=INITIAL_CAPITAL,
        transaction_cost=TRANSACTION_COST,
        target_volatility=TARGET_VOL,
    )
    df_comp = comp_suite.run_comparison(
        scores_df=scores_df,
        stock_returns=stock_returns,
        confidence_df=confidence_df,
        regime_exposure=regime_exposure,
        adv_data=adv_data,
    )

    if not df_comp.empty:
        Path("reports").mkdir(exist_ok=True)
        out_path = Path("reports/portfolio_optimizer_comparison.csv")
        df_comp.to_csv(out_path)
        logger.info(f"  [Portfolio Comparison] Saved multi-optimizer metrics to {out_path}")
        logger.info(f"\n{df_comp.to_string()}\n")

    return df_comp


def step8b_institutional_risk_audit(
    weight_df: pd.DataFrame,
    stock_panel: pd.DataFrame,
    nifty_df: pd.DataFrame,
):
    """Phase 9 Institutional Risk Engine Audit & 5 Formal Reports Generation."""
    logger.info("=" * 70)
    logger.info("§VIII.C — PHASE 9 INSTITUTIONAL RISK ENGINE AUDIT & REPORTS")
    logger.info("=" * 70)

    close_prices = stock_panel["Close"].unstack(level="Ticker").sort_index()
    stock_returns = close_prices.pct_change().fillna(0.0)
    adv_data = (stock_panel["Close"] * stock_panel["Volume"]).unstack(level=1).rolling(20, min_periods=5).mean().ffill()
    nifty_returns = nifty_df["Close"].pct_change().fillna(0.0) if "Close" in nifty_df.columns else None

    latest_weights = weight_df.iloc[-1].dropna()
    latest_adv = adv_data.iloc[-1] if not adv_data.empty else None

    risk_engine = InstitutionalRiskEngine()
    report = risk_engine.audit_portfolio_risk(
        weights=latest_weights,
        returns_df=stock_returns,
        adv_data=latest_adv,
        benchmark_returns=nifty_returns,
    )

    logger.info(f"  Historical VaR (95%)   : {report.var_95_historical:.2%}")
    logger.info(f"  Historical CVaR (95%)  : {report.cvar_95:.2%}")
    logger.info(f"  Parametric VaR (95%)   : {report.var_95_parametric:.2%}")
    logger.info(f"  Monte Carlo VaR (95%)  : {report.var_95_monte_carlo:.2%}")
    logger.info(f"  Days to Liquidate (95%): {report.days_to_liquidate_95pct:.1f} days")
    logger.info(f"  Risk Limits Passed     : {report.position_limits_passed}")
    return report


def step5_decision_matrix(
    alpha_stats:   Dict[str, Any],
    prod_results:  Dict[str, Any],
    regime_res:    Dict[str, Any],
    stress_res:    Dict[str, Any],
):
    logger.info("=" * 70)


def step6_save_reports(prod_results: Dict[str, Any]):
    Path("reports").mkdir(exist_ok=True)

    eq = prod_results["equity_curve"]
    eq.plot(title="QuantDeck CQRO — Net Equity Curve (2005–2026)", grid=True, figsize=(14, 6))
    plt.tight_layout()
    plt.savefig("reports/cqro_equity_curve.png", dpi=150)
    plt.close()

    monthly = prod_results["monthly_returns"]["Monthly Return"]
    monthly_pct = monthly.map(lambda x: f"{float(x):+.1%}" if pd.notna(x) else "N/A")
    monthly_df = pd.DataFrame({
        "Year":  monthly.index.year,
        "Month": monthly.index.month,
        "Ret":   monthly_pct.values,
    })
    pivot = monthly_df.pivot(index="Year", columns="Month", values="Ret").fillna("")
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot.columns = [month_names[m - 1] for m in pivot.columns]
    logger.info("\n" + "─" * 70)
    logger.info("MONTHLY RETURNS TABLE")
    logger.info("─" * 70)
    logger.info("\n" + pivot.to_string())

    logger.info("\n  Reports saved to ./reports/")


def step7_generate_live_orders(weight_df: pd.DataFrame, stock_panel: pd.DataFrame, initial_capital: float = INITIAL_CAPITAL):
    logger.info("=" * 70)
    logger.info("§X — LIVE ORDER GENERATOR (EXACT SHARES)")
    logger.info("=" * 70)
    
    if weight_df.empty:
        logger.warning("  No weights available to generate orders.")
        return
        
    latest_date = weight_df.index[-1]
    latest_weights = weight_df.loc[latest_date]
    active_weights = latest_weights[latest_weights > 0].sort_values(ascending=False)
    
    if active_weights.empty:
        logger.info("  No active positions for the latest target date.")
        return
        
    close_prices = stock_panel["Close"].unstack(level="Ticker")
    latest_price_date = close_prices.index[-1]
    latest_prices = close_prices.loc[latest_price_date]
    
    orders = []
    for ticker, weight in active_weights.items():
        if ticker in latest_prices and pd.notna(latest_prices[ticker]):
            price = latest_prices[ticker]
            allocated_capital = weight * initial_capital
            shares = int(allocated_capital // price)
            if shares > 0:
                orders.append({
                    "Ticker": ticker,
                    "Target_Weight_%": round(weight * 100, 2),
                    "Allocated_Capital": round(allocated_capital, 2),
                    "Latest_Price": round(price, 2),
                    "Shares_To_Buy": shares
                })
                
    if not orders:
        logger.info("  No whole shares could be allocated with the current capital.")
        return
        
    orders_df = pd.DataFrame(orders)
    
    logger.info(f"\n  LIVE ORDERS FOR {latest_date.date()} (Prices as of {latest_price_date.date()})\n")
    logger.info(orders_df.to_string(index=False))
    
    out_path = Path("reports/daily_orders.csv")
    orders_df.to_csv(out_path, index=False)
    logger.info(f"\n  ✅ Explicit Shares list saved to {out_path.resolve()}")


def step8_daily_summary(weight_df: pd.DataFrame, stock_panel: pd.DataFrame, initial_capital: float = INITIAL_CAPITAL):
    if weight_df.empty:
        logger.warning("  [Summary] No weights to summarise.")
        return
    latest_date = weight_df.index[-1]
    latest_weights = weight_df.loc[latest_date]
    active_weights = latest_weights[latest_weights > 0].sort_values(ascending=False)
    close_prices = stock_panel["Close"].unstack(level="Ticker")
    latest_price_date = close_prices.index[-1]
    latest_prices = close_prices.loc[latest_price_date]
    rows = []
    for ticker, weight in active_weights.items():
        if ticker not in latest_prices or pd.isna(latest_prices[ticker]):
            continue
        price = latest_prices[ticker]
        allocated_cap = weight * initial_capital
        shares = int(allocated_cap // price)
        if shares <= 0:
            continue
        stop_loss = round(price * 0.98, 2)
        target = round(price * 1.05, 2)
        rows.append({
            "Date": latest_date.date(),
            "Ticker": ticker,
            "Weight_%": round(weight * 100, 2),
            "Allocated_Capital": round(allocated_cap, 2),
            "Entry_Price": round(price, 2),
            "Shares": shares,
            "Stop_Loss": stop_loss,
            "Target": target,
        })
    if not rows:
        logger.info("  [Summary] No share allocations after rounding.")
        return
    summary_df = pd.DataFrame(rows)
    out_path = Path("reports/daily_summary.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_path, index=False)
    logger.info(f"\n  📊 Daily allocation summary saved to {out_path.resolve()}")


def step9_cleanup_reports():
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    keep_files = {
        "daily_orders.csv", "daily_summary.csv", "prediction_explanations.csv",
        "confidence_report.csv", "drift_report.json", "drift_dashboard.json",
        "monitoring_health_report.json", "cqro_equity_curve.png"
    }
    for f in reports_dir.glob("*.csv"):
        if f.name not in keep_files:
            try:
                f.unlink()
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
# §X — INSTITUTIONAL MONITORING PLATFORM (Phases 6 & 7 Integration)
# ══════════════════════════════════════════════════════════════════════════════
def step10_monitoring(weight_df, stock_panel, feature_panel, prod_results, scores_df):
    import json
    logger.info("=" * 70)
    logger.info("§X — INSTITUTIONAL MONITORING PLATFORM")
    logger.info("=" * 70)

    try:
        mon = MonitoringLayer()

        close_df = stock_panel["Close"].unstack(level="Ticker")
        mon.record_feed_update("price_feed", data=close_df)
        mon.record_feed_update("feature_feed", n_rows=len(close_df))
        mon.record_feed_update("prediction_feed", n_rows=len(weight_df))
        mon.record_feed_update("fundamental_feed", n_rows=len(stock_panel))

        # Register model & predictions for ModelHealthMonitor
        mon.record_model_training("EnsembleAlphaModel", "1.0.0")
        if not scores_history_array(scores_df).size == 0:
            mon.record_predictions(
                model_id="EnsembleAlphaModel",
                model_version="1.0.0",
                scores=scores_history_array(scores_df),
            )

        equity_curve = None
        daily_returns = None
        market_returns = None
        if "equity_curve" in prod_results:
            equity_curve = prod_results["equity_curve"]
            daily_returns = equity_curve.pct_change().dropna()
        elif "daily_returns" in prod_results:
            daily_returns = prod_results["daily_returns"]

        if close_df.shape[1] > 0:
            market_returns = close_df.pct_change().mean(axis=1).dropna()

        latest_weights = None
        if not weight_df.empty:
            latest_weights = weight_df.iloc[-1]
            latest_weights = latest_weights[latest_weights.abs() > 1e-6]

        # Prepare reference & current feature panels for Phase 6 Drift Detection
        ref_df = feature_panel.head(len(feature_panel) // 2)
        cur_df = feature_panel.tail(len(feature_panel) // 2)

        health_report = mon.full_health_check(
            data_df=close_df,
            daily_returns=daily_returns,
            equity_curve=equity_curve,
            market_returns=market_returns,
            portfolio_weights=latest_weights,
            returns_df=close_df.pct_change().dropna() if close_df is not None else None,
            reference_df=ref_df,
            current_df=cur_df,
        )

        out_path = Path("reports/monitoring_health_report.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {}
        for k, v in health_report.items():
            if isinstance(v, dict):
                serializable[k] = {
                    sk: (sv if not isinstance(sv, (pd.Series, pd.DataFrame, np.ndarray)) else str(type(sv).__name__))
                    for sk, sv in v.items()
                }
            elif isinstance(v, (pd.Series, pd.DataFrame, np.ndarray)):
                serializable[k] = str(type(v).__name__)
            else:
                serializable[k] = v
        out_path.write_text(json.dumps(serializable, indent=2, default=str))

        overall = health_report.get("overall_health", "UNKNOWN")
        logger.info(f"  📊 Overall System Health: {overall}")
        logger.info(f"  📊 Monitoring report saved to {out_path.resolve()}")

        try:
            mon.dashboard._rich = None
            mon.render_dashboard()
        except Exception as dash_exc:
            logger.warning(f"[Monitoring] Dashboard render skipped: {dash_exc}")

    except Exception as exc:
        logger.warning(f"[Monitoring] Non-fatal monitoring error: {exc}")


def scores_history_array(scores_df: pd.DataFrame) -> np.ndarray:
    if scores_df.empty:
        return np.array([])
    return scores_df.iloc[-1].dropna().values


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════════════
def main():
    logger.info("QuantDeck — CQRO Institutional Alpha Engine Starting")
    logger.info("=" * 70)

    # §I   Data
    stock_panel, nifty_df, vix_df = step1_fetch_data()

    # §II  Features
    feature_panel = step2_build_features(stock_panel)
    full_panel    = build_target_panel(feature_panel, price_col="Close", horizon=REBALANCE_HORIZON)

    # §III Walk-Forward (with Persistence)
    cache_path = Path("data_cache/walk_forward_cache.pkl")
    latest_model = None
    if cache_path.exists():
        logger.info(">>> LOADING CACHED WALK-FORWARD RESULTS (Step III Skip) <<<")
        import pickle
        with open(cache_path, "rb") as f:
            weight_df, scores_df = pickle.load(f)
    else:
        weight_df, scores_df, latest_model = step3_walk_forward(full_panel)
        import pickle
        with open(cache_path, "wb") as f:
            pickle.dump((weight_df, scores_df), f)
        logger.info(f"  Step III results cached to {cache_path}")

    # §III.B Explainability & Confidence
    if latest_model is not None:
        step3b_explainability_and_confidence(latest_model, full_panel)

    # §IV  Pure Alpha Validation
    alpha_stats = evaluate_pure_alpha(scores_df, stock_panel, transaction_cost=TRANSACTION_COST, initial_capital=INITIAL_CAPITAL)

    # §V/VI Production Backtest
    prod_results = step4_production_backtest(weight_df, stock_panel, nifty_df, vix_df, alpha_stats.get("ic_series", pd.Series()))

    # §VIII.B Phase 8 Multi-Optimizer Comparison
    step4b_portfolio_comparison(scores_df, stock_panel, nifty_df, vix_df)

    # §VII Regime Robustness
    regime_exposure = compute_regime_exposure(nifty_df, vix_df=vix_df)
    regime_res = run_regime_robustness(
        ic_series      = alpha_stats.get("ic_series", pd.Series(dtype=float)),
        equity_curve   = prod_results["equity_curve"],
        daily_returns  = prod_results["daily_returns"],
        nifty_df       = nifty_df,
        regime_exposure= regime_exposure,
    )

    # §VIII Stress Tests
    adv_data = (stock_panel["Close"] * stock_panel["Volume"]).unstack(level=1).rolling(20, min_periods=5).mean().ffill()
    stress_res = run_stress_tests(
        weight_df       = weight_df,
        stock_panel     = stock_panel,
        regime_exposure = regime_exposure,
        adv_data        = adv_data,
        base_sharpe     = prod_results["sharpe_ratio"],
        transaction_cost= TRANSACTION_COST,
        impact_coeff    = IMPACT_COEFF,
        initial_capital = INITIAL_CAPITAL,
    )

    # §VIII.C Phase 9 Institutional Risk Engine Audit & Reports
    step8b_institutional_risk_audit(weight_df, stock_panel, nifty_df)

    # §IX  Decision Matrix
    step5_decision_matrix(alpha_stats, prod_results, regime_res, stress_res)

    # Save reports
    step6_save_reports(prod_results)

    # Generate exact shares for execution
    step7_generate_live_orders(weight_df, stock_panel, INITIAL_CAPITAL)

    step8_daily_summary(weight_df, stock_panel, INITIAL_CAPITAL)
    step10_monitoring(weight_df, stock_panel, feature_panel, prod_results, scores_df)
    step9_cleanup_reports()

    logger.info("CQRO Engine Run Complete.")


if __name__ == "__main__":
    main()

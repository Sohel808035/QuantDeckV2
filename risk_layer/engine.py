"""
risk_layer/engine.py
────────────────────
Master Institutional Risk Engine Orchestrator for Phase 9.
Combines Historical/Parametric/Monte Carlo VaR/CVaR, Stress Testing, Scenario Analysis,
Liquidity Risk, Factor Risk (Momentum, Volatility, Value, Size, Quality), Beta Exposure,
Sector/Industry Exposures, Correlation Risk, Heatmaps, and Limits Auditing into a single API.
Generates 5 formal institutional risk reports.
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Set, Any, List
import pandas as pd
import numpy as np

from risk_layer.base import RiskMetricsReport
from risk_layer.config import RiskConfig
from risk_layer.var_cvar import VaRCVaREngine
from risk_layer.stress_testing import StressTestingEngine
from risk_layer.liquidity_risk import LiquidityRiskEngine
from risk_layer.factor_risk import FactorRiskEngine
from risk_layer.sector_country_exposure import ExposureRiskEngine
from risk_layer.correlation_analysis import CorrelationAnalysisEngine
from risk_layer.tail_risk import TailRiskEngine
from risk_layer.scenario_analysis import ScenarioAnalysisEngine
from risk_layer.limits import LimitsAuditEngine
from risk_layer.heatmaps import RiskHeatmapEngine

logger = logging.getLogger(__name__)


class InstitutionalRiskEngine:
    """Master Institutional Risk Management Suite."""

    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()
        self.var_engine = VaRCVaREngine(confidence_levels=self.config.confidence_levels)
        self.stress_engine = StressTestingEngine()
        self.liquidity_engine = LiquidityRiskEngine(max_adv_participation=self.config.max_adv_participation_pct)
        self.factor_engine = FactorRiskEngine()
        self.exposure_engine = ExposureRiskEngine()
        self.correlation_engine = CorrelationAnalysisEngine()
        self.tail_engine = TailRiskEngine(evt_quantile=self.config.evt_threshold_quantile)
        self.scenario_engine = ScenarioAnalysisEngine()
        self.limits_engine = LimitsAuditEngine(config=self.config)
        self.heatmap_engine = RiskHeatmapEngine()

    def audit_portfolio_risk(
        self,
        weights: pd.Series,
        returns_df: Optional[pd.DataFrame] = None,
        adv_data: Optional[pd.Series] = None,
        sector_map: Optional[Dict[str, str]] = None,
        country_map: Optional[Dict[str, str]] = None,
        industry_map: Optional[Dict[str, str]] = None,
        stock_betas: Optional[pd.Series] = None,
        factor_beta_matrix: Optional[pd.DataFrame] = None,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> RiskMetricsReport:
        """
        Executes comprehensive 13-factor institutional risk audit across portfolio positions.
        """
        if weights.empty:
            return RiskMetricsReport(portfolio_value=self.config.portfolio_value)

        w = weights / weights.sum()
        tickers = list(w.index)
        adv_series = adv_data if adv_data is not None else pd.Series(dtype=float)

        # Portfolio return series
        if returns_df is not None and not returns_df.empty:
            common = list(set(tickers) & set(returns_df.columns))
            if common:
                port_returns = (returns_df[common].dropna(how="all") * w.reindex(common).fillna(0)).sum(axis=1)
            else:
                port_returns = pd.Series(dtype=float)
        else:
            port_returns = pd.Series(dtype=float)

        # 1. VaR & CVaR
        var95_h, cvar95 = self.var_engine.historical_var_cvar(port_returns, confidence=0.95)
        var99_h, cvar99 = self.var_engine.historical_var_cvar(port_returns, confidence=0.99)
        var95_p, _      = self.var_engine.parametric_var_cvar(port_returns, confidence=0.95)
        var99_p, _      = self.var_engine.parametric_var_cvar(port_returns, confidence=0.99)
        var95_mc, _     = self.var_engine.monte_carlo_var_cvar(port_returns, confidence=0.95)

        # 2. Liquidity Risk & LVaR
        dtl = self.liquidity_engine.portfolio_days_to_liquidate(w, adv_series, self.config.portfolio_value)
        lvar95 = self.liquidity_engine.liquidity_adjusted_var(var95_h, w, adv_series, self.config.portfolio_value)

        # 3. Concentration & Limits Auditing
        conc = self.limits_engine.concentration_metrics(w)
        passed, checks_dict, warnings_list = self.limits_engine.audit_limits(
            w,
            sector_map=sector_map,
            industry_map=industry_map,
            var_95=var95_h,
            cvar_95=cvar95,
            days_to_liquidate=dtl,
        )

        for w_item in warnings_list:
            logger.warning(f"[RISK LIMIT ALERT] [{w_item['category']}] {w_item['message']}")

        # 4. Tail Risk & Correlation
        tail_metrics = self.tail_engine.compute_tail_metrics(port_returns)
        corr_metrics = self.correlation_engine.compute_correlation_metrics(returns_df if returns_df is not None else pd.DataFrame(), tickers=tickers)

        # 5. Sector, Country & Factor Exposure
        sec_df = self.exposure_engine.compute_sector_exposure(w, sector_map or {})
        sec_dict = sec_df["portfolio_weight"].to_dict() if not sec_df.empty else {}
        cntry_series = self.exposure_engine.compute_country_exposure(w, country_map)
        cntry_dict = cntry_series.to_dict()

        factor_exp_s = self.factor_engine.compute_factor_exposures(w, factor_beta_matrix, returns_df)
        factor_exp = factor_exp_s.to_dict()

        # 6. Beta Exposure
        port_beta = 1.0
        if benchmark_returns is not None and returns_df is not None:
            port_beta = self.factor_engine.compute_portfolio_beta(w, returns_df, benchmark_returns)

        # 7. Stress Testing & Scenario Analysis
        stress_losses = self.stress_engine.run_historical_replay(w, stock_betas=stock_betas, portfolio_value=self.config.portfolio_value)
        scenario_losses = self.scenario_engine.run_scenario_matrix(w, stock_betas=stock_betas, portfolio_value=self.config.portfolio_value)

        # 8. Portfolio Heatmaps
        heatmaps = {}
        if returns_df is not None and not returns_df.empty:
            heatmaps = self.heatmap_engine.compute_risk_heatmaps(w, returns_df, sector_map)

        report = RiskMetricsReport(
            portfolio_value=self.config.portfolio_value,
            var_95_historical=var95_h,
            var_99_historical=var99_h,
            var_95_parametric=var95_p,
            var_99_parametric=var99_p,
            var_95_monte_carlo=var95_mc,
            cvar_95=cvar95,
            cvar_99=cvar99,
            lvar_95=lvar95,
            hhi_index=conc["hhi_index"],
            effective_n_stocks=conc["effective_n_stocks"],
            top_5_concentration=conc["top_5_concentration"],
            top_10_concentration=conc["top_10_concentration"],
            max_position_weight=float(w.max()),
            position_limits_passed=passed,
            skewness=tail_metrics["skewness"],
            kurtosis=tail_metrics["kurtosis"],
            evt_tail_index=tail_metrics["evt_tail_index"],
            avg_pairwise_correlation=corr_metrics["avg_pairwise_correlation"],
            pca_top3_variance_pct=corr_metrics["pca_top3_var"],
            days_to_liquidate_95pct=dtl,
            sector_exposures=sec_dict,
            country_exposures=cntry_dict,
            factor_exposures=factor_exp,
            stress_test_losses=stress_losses,
            scenario_impacts=scenario_losses,
        )

        # Generate the 5 formal institutional reports
        self.generate_all_reports(report, warnings_list, heatmaps, dtl, conc)

        return report

    def generate_all_reports(
        self,
        report: RiskMetricsReport,
        warnings: List[Dict[str, Any]],
        heatmaps: Dict[str, Any],
        dtl: float,
        conc: Dict[str, float],
        output_dir: str = "reports",
    ) -> Dict[str, str]:
        """Generates the 5 formal institutional risk reports."""
        out = Path(output_dir)
        out.mkdir(exist_ok=True)

        paths = {
            "risk_report": out / "risk_report.json",
            "exposure_report": out / "exposure_report.json",
            "var_report": out / "var_report.json",
            "stress_test_report": out / "stress_test_report.json",
            "liquidity_report": out / "liquidity_report.json",
        }

        # 1. Master Risk Report
        risk_data = {
            "portfolio_value": report.portfolio_value,
            "limits_passed": report.position_limits_passed,
            "warnings": warnings,
            "concentration": conc,
            "tail_risk": {
                "skewness": report.skewness,
                "kurtosis": report.kurtosis,
                "evt_tail_index": report.evt_tail_index,
            },
            "correlation_risk": {
                "avg_pairwise_correlation": report.avg_pairwise_correlation,
                "pca_top3_variance_pct": report.pca_top3_variance_pct,
            },
        }

        # 2. Exposure Report
        exposure_data = {
            "sector_exposures": report.sector_exposures,
            "country_exposures": report.country_exposures,
            "factor_exposures": report.factor_exposures,
            "max_position_weight": report.max_position_weight,
            "heatmaps": heatmaps,
        }

        # 3. VaR Report
        var_data = {
            "var_95_historical": report.var_95_historical,
            "var_99_historical": report.var_99_historical,
            "var_95_parametric": report.var_95_parametric,
            "var_99_parametric": report.var_99_parametric,
            "var_95_monte_carlo": report.var_95_monte_carlo,
            "cvar_95": report.cvar_95,
            "cvar_99": report.cvar_99,
            "lvar_95": report.lvar_95,
        }

        # 4. Stress Test Report
        stress_test_data = {
            "historical_stress_replay": report.stress_test_losses,
            "scenario_shocks": report.scenario_impacts,
        }

        # 5. Liquidity Report
        liquidity_data = {
            "days_to_liquidate_95pct": dtl,
            "max_adv_participation": self.config.max_adv_participation_pct,
            "lvar_95": report.lvar_95,
        }

        for key, p in paths.items():
            with open(p, "w") as f:
                json.dump(locals()[key.replace("_report", "") + "_data"], f, indent=2)

        logger.info(f"  [Institutional Risk Engine] 5 formal reports written to {out}")
        return {k: str(v) for k, v in paths.items()}

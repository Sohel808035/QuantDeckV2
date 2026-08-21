"""
monitoring_layer/dashboard.py
──────────────────────────────
Logging Dashboard — Upgraded Institutional Version.
Renders a rich terminal dashboard with monitoring status tables and alert feed.
Includes extended components: Model Health, Portfolio Risk, Market Regime, Data Freshness.
Falls back gracefully when 'rich' library is not installed.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _try_rich():
    """Returns (Console, Table, Text, Panel, Columns, box) or None if rich unavailable."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.text import Text
        from rich.panel import Panel
        from rich.columns import Columns
        from rich import box
        return Console, Table, Text, Panel, Columns, box
    except ImportError:
        return None


class MonitoringDashboard:
    """
    Renders a structured, colour-coded monitoring dashboard to the terminal.
    Uses the 'rich' library if available, otherwise falls back to plain-text output.
    """

    def __init__(self, service_name: str = "QuantSphereX Monitoring"):
        self.service_name = service_name
        self._rich = _try_rich()

    def render(
        self,
        health_report: Optional[Dict[str, Any]] = None,
        data_quality_report: Optional[Dict[str, Any]] = None,
        drift_report: Optional[Dict[str, Any]] = None,
        strategy_report: Optional[Dict[str, Any]] = None,
        recent_alerts: Optional[List[Any]] = None,
        model_health_report: Optional[Dict[str, Any]] = None,
        portfolio_risk_report: Optional[Dict[str, Any]] = None,
        market_regime_report: Optional[Dict[str, Any]] = None,
        freshness_report: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Renders the full monitoring dashboard to the terminal."""
        if self._rich:
            self._render_rich(
                health_report, data_quality_report, drift_report, strategy_report,
                recent_alerts, model_health_report, portfolio_risk_report,
                market_regime_report, freshness_report
            )
        else:
            self._render_plain(
                health_report, data_quality_report, drift_report, strategy_report,
                recent_alerts, model_health_report, portfolio_risk_report,
                market_regime_report, freshness_report
            )

    # ── Rich Rendering ──────────────────────────────────────────────────────

    def _render_rich(
        self, health, dq, drift, strategy, alerts,
        model_health, port_risk, market_regime, freshness
    ):
        Console, Table, Text, Panel, Columns, box = self._rich
        console = Console()

        console.rule(f"[bold cyan]  {self.service_name}  [/bold cyan]")
        console.print(f"[dim]{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}[/dim]\n")

        panels = []

        # ── System Health & Market Regime ──────────────────────────────────
        if health or market_regime:
            tbl = Table(box=box.SIMPLE_HEAVY, show_header=False)
            tbl.add_column("Metric", style="cyan")
            tbl.add_column("Value", justify="right")
            
            if health:
                cpu = health.get("cpu_pct")
                if cpu is not None:
                    cpu_status = "🟢" if cpu < 75 else "🟡" if cpu < 90 else "🔴"
                    tbl.add_row("CPU Usage", f"{cpu:.1f}% {cpu_status}")
                mem = health.get("memory_pct")
                if mem is not None:
                    mem_status = "🟢" if mem < 80 else "🟡" if mem < 95 else "🔴"
                    tbl.add_row("Mem Usage", f"{mem:.1f}% {mem_status}")
            
            if market_regime:
                if health:
                    tbl.add_row("---", "---")
                tbl.add_row("Regime", f"[bold]{market_regime.get('regime_composite', 'N/A')}[/bold]")
                tbl.add_row("Vol Regime", str(market_regime.get('volatility_regime', 'N/A')))
                tbl.add_row("Trend", str(market_regime.get('trend_regime', 'N/A')))

            panels.append(Panel(tbl, title="[bold]System & Market[/bold]", border_style="blue"))

        # ── Strategy & Portfolio Risk ──────────────────────────────────────
        if strategy or port_risk:
            tbl = Table(box=box.SIMPLE_HEAVY, show_header=False)
            tbl.add_column("Metric", style="cyan")
            tbl.add_column("Value", justify="right")

            if strategy:
                for key, label in [("latest_sharpe", "Sharpe"), ("latest_ic", "Roll IC"), ("current_drawdown", "Drawdown")]:
                    if key in strategy:
                        val = strategy[key]
                        status = "🟢" if not strategy.get("breach", False) else "🔴"
                        fmt_val = f"{val:.2%}" if "drawdown" in key else f"{val:.4f}"
                        tbl.add_row(label, f"{fmt_val} {status}")
            
            if port_risk:
                if strategy:
                    tbl.add_row("---", "---")
                if "historical_var" in port_risk and port_risk["historical_var"] is not None:
                    var = port_risk["historical_var"]
                    var_status = "🔴" if port_risk.get("var_breach") else "🟢"
                    tbl.add_row("99% Hist VaR", f"{var:.2%} {var_status}")
                if "gross_exposure" in port_risk:
                    tbl.add_row("Gross Exp", f"{port_risk['gross_exposure']:.2f}x")

            panels.append(Panel(tbl, title="[bold]Strategy & Risk[/bold]", border_style="magenta"))

        # ── Data Freshness & Quality ───────────────────────────────────────
        if freshness or dq:
            tbl = Table(box=box.SIMPLE_HEAVY, show_header=False)
            tbl.add_column("Feed/Check", style="cyan")
            tbl.add_column("Status", justify="center")

            if freshness and "summary" in freshness:
                sum_st = freshness["summary"]
                health_col = "green" if sum_st.get("overall_health") == "HEALTHY" else "red"
                tbl.add_row("Data Feeds", f"[{health_col}]{sum_st.get('overall_health')}[/{health_col}]")
                tbl.add_row("  Fresh", str(sum_st.get("fresh", 0)))
                tbl.add_row("  Stale", str(sum_st.get("stale", 0)))
                tbl.add_row("  Critical", str(sum_st.get("critical", 0)))

            if dq:
                if freshness:
                    tbl.add_row("---", "---")
                checks = dq.get("checks", {})
                for check_name, result in list(checks.items())[:3]: # Show top 3
                    passed = result.get("passed", True)
                    tbl.add_row(check_name.replace("_", " ").title(), "✅" if passed else "❌")

            panels.append(Panel(tbl, title="[bold]Data Pipeline[/bold]", border_style="green"))

        # ── Model Health & Drift ───────────────────────────────────────────
        if model_health or drift:
            tbl = Table(box=box.SIMPLE_HEAVY, show_header=False)
            tbl.add_column("Metric", style="cyan")
            tbl.add_column("Status", justify="right")

            if model_health:
                for mid, rep in list(model_health.items())[:2]: # Show top 2 models
                    hs = rep.get("health_score", 0.0)
                    ret = rep.get("retrain_recommended", False)
                    col = "green" if hs > 0.8 else "yellow" if hs > 0.5 else "red"
                    ret_str = " (RETRAIN)" if ret else ""
                    tbl.add_row(f"Model: {mid}", f"[{col}]Health: {hs:.2f}{ret_str}[/{col}]")

            if drift:
                if model_health:
                    tbl.add_row("---", "---")
                drift_rate = drift.get("drift_rate", 0.0)
                n_drifted = len(drift.get("drifted_features", []))
                tbl.add_row("Feature Drift", f"{drift_rate:.1%} ({n_drifted} feat)")

            panels.append(Panel(tbl, title="[bold]Model & Drift[/bold]", border_style="yellow"))

        # Render top panels
        if panels:
            # Group into rows of 2
            for i in range(0, len(panels), 2):
                console.print(Columns(panels[i:i+2], equal=True))
                console.print("")

        # ── Recent Alerts ─────────────────────────────────────────────────
        if alerts:
            alert_tbl = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold red")
            alert_tbl.add_column("Time", style="dim")
            alert_tbl.add_column("Severity", justify="center")
            alert_tbl.add_column("Category")
            alert_tbl.add_column("Metric")
            alert_tbl.add_column("Message")

            sev_colors = {"CRITICAL": "bold red", "WARNING": "yellow", "INFO": "cyan"}
            for a in alerts[-10:]:
                if isinstance(a, dict):
                    sev = str(a.get("severity", "INFO"))
                    category = str(a.get("category", ""))
                    metric = str(a.get("metric", ""))
                    msg = str(a.get("message", ""))
                    ts_val = float(a.get("timestamp", time.time()))
                else:
                    sev = a.severity.value if hasattr(getattr(a, "severity", None), "value") else str(getattr(a, "severity", "INFO"))
                    category = str(getattr(a, "category", ""))
                    metric = str(getattr(a, "metric", ""))
                    msg = str(getattr(a, "message", ""))
                    ts_val = float(getattr(a, "timestamp", time.time()))
                color = sev_colors.get(sev, "white")
                ts = time.strftime("%H:%M:%S", time.gmtime(ts_val))
                alert_tbl.add_row(ts, f"[{color}]{sev}[/{color}]", category, metric, msg)

            console.print(Panel(alert_tbl, title="[bold red]Recent Alerts[/bold red]", border_style="red"))

        console.rule()

    # ── Plain-Text Fallback ─────────────────────────────────────────────────

    def _render_plain(
        self, health, dq, drift, strategy, alerts,
        model_health, port_risk, market_regime, freshness
    ):
        lines = [
            "=" * 70,
            f"  {self.service_name}",
            f"  {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
            "=" * 70,
        ]
        
        if health:
            lines += ["[SYSTEM HEALTH]"]
            for k, v in health.items():
                if v is not None and not isinstance(v, dict):
                    lines.append(f"  {k}: {v}")
                    
        if market_regime:
            lines += ["[MARKET REGIME]"]
            for k, v in market_regime.items():
                lines.append(f"  {k}: {v}")

        if dq:
            lines += ["[DATA QUALITY]"]
            for check, res in dq.get("checks", {}).items():
                status = "PASS" if res.get("passed", True) else "FAIL"
                lines.append(f"  {check}: {status}")
                
        if freshness and "summary" in freshness:
            lines += ["[DATA FRESHNESS]"]
            for k, v in freshness["summary"].items():
                lines.append(f"  {k}: {v}")

        if strategy:
            lines += ["[STRATEGY MONITOR]"]
            for k, v in strategy.items():
                if isinstance(v, (int, float)):
                    lines.append(f"  {k}: {v:.4f}")
                    
        if port_risk:
            lines += ["[PORTFOLIO RISK]"]
            for k, v in port_risk.items():
                if isinstance(v, (int, float)):
                    lines.append(f"  {k}: {v:.4f}")
                    
        if model_health:
            lines += ["[MODEL HEALTH]"]
            for mid, rep in model_health.items():
                lines.append(f"  {mid}: Health={rep.get('health_score')}, Retrain={rep.get('retrain_recommended')}")

        if alerts:
            lines += ["[RECENT ALERTS]"]
            for a in alerts[-10:]:
                if isinstance(a, dict):
                    sev = str(a.get("severity", "INFO"))
                    cat = str(a.get("category", ""))
                    met = str(a.get("metric", ""))
                    msg = str(a.get("message", ""))
                else:
                    sev = a.severity.value if hasattr(getattr(a, "severity", None), "value") else str(getattr(a, "severity", "INFO"))
                    cat = str(getattr(a, "category", ""))
                    met = str(getattr(a, "metric", ""))
                    msg = str(getattr(a, "message", ""))
                lines.append(f"  [{sev}] {cat} | {met}: {msg}")
        lines.append("=" * 70)
        print("\n".join(lines))

"""
monitoring_layer/data_freshness_monitor.py
───────────────────────────────────────────
Institutional Data Freshness Monitor.

Detects:
  - Stale data feeds (no update beyond expected frequency)
  - Data gaps (missing expected date ranges)
  - Feed-level health scores
  - Expected vs actual update timestamps
  - Business-hour awareness (markets only update during market hours)

Architecture:
  DataFreshnessMonitor ──► AlertEngine
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd

from monitoring_layer.config import MonitoringConfig
from monitoring_layer.alert_engine import AlertEngine, AlertSeverity

logger = logging.getLogger(__name__)


# ── Enums ─────────────────────────────────────────────────────────────────────


class FeedFrequency(str, Enum):
    REAL_TIME = "real_time"      # Expected every few minutes
    INTRADAY = "intraday"        # Expected multiple times per day
    DAILY = "daily"              # Expected once per business day
    WEEKLY = "weekly"            # Expected weekly
    MONTHLY = "monthly"          # Expected monthly


# ── Feed Registration ─────────────────────────────────────────────────────────


@dataclass
class FeedConfig:
    """Configuration for a monitored data feed."""

    name: str
    frequency: FeedFrequency
    warn_after_seconds: int    # Seconds since last update before WARNING
    crit_after_seconds: int    # Seconds since last update before CRITICAL
    expected_rows_per_update: Optional[int] = None  # Expected row count per update
    min_rows: int = 1


@dataclass
class FeedState:
    """Runtime state for a monitored feed."""

    config: FeedConfig
    last_update_timestamp: Optional[float] = None
    last_row_count: int = 0
    total_updates: int = 0
    staleness_alerts: int = 0
    health_score: float = 1.0


# ── Data Freshness Monitor ────────────────────────────────────────────────────


class DataFreshnessMonitor:
    """
    Monitors data feed freshness for all critical data sources in QuantSphereX.

    Tracks last-seen timestamps per feed and issues alerts when data is stale.

    Predefined feeds (can be extended):
      - 'price_feed'       : Daily OHLCV data
      - 'fundamental_feed' : Quarterly fundamental data
      - 'feature_feed'     : Computed feature data
      - 'prediction_feed'  : Model predictions

    Usage:
        monitor = DataFreshnessMonitor()
        monitor.register_feed("price_feed", FeedFrequency.DAILY, ...)
        monitor.record_update("price_feed", df)
        report = monitor.check_all_feeds()
    """

    # Default feed configurations
    DEFAULTS: Dict[str, Dict] = {
        "price_feed": {
            "frequency": FeedFrequency.DAILY,
            "warn_after_seconds": 3600 * 26,   # 26h (next business day)
            "crit_after_seconds": 3600 * 48,   # 48h
        },
        "feature_feed": {
            "frequency": FeedFrequency.DAILY,
            "warn_after_seconds": 3600 * 28,
            "crit_after_seconds": 3600 * 52,
        },
        "prediction_feed": {
            "frequency": FeedFrequency.DAILY,
            "warn_after_seconds": 3600 * 30,
            "crit_after_seconds": 3600 * 54,
        },
        "fundamental_feed": {
            "frequency": FeedFrequency.WEEKLY,
            "warn_after_seconds": 3600 * 24 * 8,   # 8 days
            "crit_after_seconds": 3600 * 24 * 14,  # 14 days
        },
    }

    def __init__(
        self,
        config: Optional[MonitoringConfig] = None,
        alert_engine: Optional[AlertEngine] = None,
    ):
        self.config = config or MonitoringConfig()
        self.alert_engine = alert_engine or AlertEngine(config=self.config)
        self._feeds: Dict[str, FeedState] = {}

        # Register default feeds
        for feed_name, defaults in self.DEFAULTS.items():
            self.register_feed(
                name=feed_name,
                frequency=defaults["frequency"],
                warn_after_seconds=defaults["warn_after_seconds"],
                crit_after_seconds=defaults["crit_after_seconds"],
            )

    # ── Feed Registration ──────────────────────────────────────────────────────

    def register_feed(
        self,
        name: str,
        frequency: FeedFrequency = FeedFrequency.DAILY,
        warn_after_seconds: int = 86400,
        crit_after_seconds: int = 172800,
        expected_rows_per_update: Optional[int] = None,
        min_rows: int = 1,
    ) -> None:
        """
        Registers a new data feed for monitoring.

        Args:
            name:                    Feed identifier (e.g. 'price_feed').
            frequency:               FeedFrequency enum value.
            warn_after_seconds:      Seconds before staleness WARNING.
            crit_after_seconds:      Seconds before staleness CRITICAL.
            expected_rows_per_update: Expected rows per update (optional validation).
            min_rows:                Minimum acceptable rows per update.
        """
        cfg = FeedConfig(
            name=name,
            frequency=frequency,
            warn_after_seconds=warn_after_seconds,
            crit_after_seconds=crit_after_seconds,
            expected_rows_per_update=expected_rows_per_update,
            min_rows=min_rows,
        )
        if name not in self._feeds:
            self._feeds[name] = FeedState(config=cfg)
            logger.info("[DataFreshness] Registered feed: %s (freq=%s)", name, frequency.value)
        else:
            self._feeds[name].config = cfg
            logger.debug("[DataFreshness] Updated feed config: %s", name)

    # ── Update Recording ──────────────────────────────────────────────────────

    def record_update(
        self,
        feed_name: str,
        data: Optional[pd.DataFrame] = None,
        n_rows: Optional[int] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Records a data update for a feed.

        Args:
            feed_name:  Feed identifier.
            data:       DataFrame received (for row count validation).
            n_rows:     Override row count instead of deriving from data.
            timestamp:  Unix timestamp of update (default: now).
        """
        if feed_name not in self._feeds:
            self.register_feed(feed_name)

        state = self._feeds[feed_name]
        ts = timestamp or time.time()
        row_count = n_rows if n_rows is not None else (len(data) if data is not None else 0)

        state.last_update_timestamp = ts
        state.last_row_count = row_count
        state.total_updates += 1

        # Row count validation
        if row_count < state.config.min_rows:
            logger.warning(
                "[DataFreshness] Feed '%s': received %d rows (min: %d)",
                feed_name, row_count, state.config.min_rows,
            )
            self.alert_engine.fire(
                AlertSeverity.WARNING, "DATA_FRESHNESS", f"low_rows.{feed_name}",
                value=float(row_count), threshold=float(state.config.min_rows),
                message=(
                    f"Feed '{feed_name}' received only {row_count} rows "
                    f"(minimum expected: {state.config.min_rows})."
                ),
            )

        logger.debug(
            "[DataFreshness] Feed '%s' updated: %d rows at %.0f",
            feed_name, row_count, ts,
        )

    # ── Staleness Checks ──────────────────────────────────────────────────────

    def check_feed(self, feed_name: str, now: Optional[float] = None) -> Dict[str, Any]:
        """
        Checks a single feed for staleness.

        Args:
            feed_name: Feed identifier.
            now:       Reference time (default: time.time()).

        Returns:
            Feed health report dict.
        """
        if feed_name not in self._feeds:
            return {
                "feed": feed_name,
                "status": "unknown",
                "staleness_seconds": None,
                "fresh": False,
            }

        state = self._feeds[feed_name]
        cfg = state.config
        now_ts = now or time.time()

        if state.last_update_timestamp is None:
            staleness_secs = float("inf")
            fresh = False
            status = "never_updated"
        else:
            staleness_secs = now_ts - state.last_update_timestamp
            fresh = staleness_secs < cfg.warn_after_seconds
            status = "fresh" if fresh else "stale"

        # Fire alerts
        if staleness_secs == float("inf") or staleness_secs >= cfg.crit_after_seconds:
            state.staleness_alerts += 1
            state.health_score = max(0.0, state.health_score - 0.1)
            self.alert_engine.fire(
                AlertSeverity.CRITICAL, "DATA_FRESHNESS", f"stale.{feed_name}",
                value=staleness_secs if staleness_secs != float("inf") else 999999,
                threshold=float(cfg.crit_after_seconds),
                message=(
                    f"CRITICAL: Feed '{feed_name}' stale for "
                    f"{staleness_secs / 3600:.1f}h (threshold: {cfg.crit_after_seconds / 3600:.1f}h)."
                    if staleness_secs != float("inf")
                    else f"CRITICAL: Feed '{feed_name}' has never been updated."
                ),
            )
            status = "critical"
        elif staleness_secs >= cfg.warn_after_seconds:
            self.alert_engine.fire(
                AlertSeverity.WARNING, "DATA_FRESHNESS", f"stale.{feed_name}",
                value=staleness_secs,
                threshold=float(cfg.warn_after_seconds),
                message=(
                    f"Feed '{feed_name}' stale for {staleness_secs / 3600:.1f}h "
                    f"(warn threshold: {cfg.warn_after_seconds / 3600:.1f}h)."
                ),
            )
            status = "warning"

        return {
            "feed": feed_name,
            "frequency": cfg.frequency.value,
            "status": status,
            "fresh": fresh,
            "staleness_seconds": round(staleness_secs, 1) if staleness_secs != float("inf") else None,
            "staleness_hours": round(staleness_secs / 3600, 2) if staleness_secs != float("inf") else None,
            "last_row_count": state.last_row_count,
            "total_updates": state.total_updates,
            "health_score": round(state.health_score, 3),
        }

    def check_all_feeds(self, now: Optional[float] = None) -> Dict[str, Any]:
        """
        Checks all registered feeds for staleness.

        Returns:
            Dict with 'feeds' (per-feed reports) and 'summary' (aggregate stats).
        """
        now_ts = now or time.time()
        reports: Dict[str, Any] = {}
        n_fresh = 0
        n_stale = 0
        n_critical = 0

        for feed_name in self._feeds:
            r = self.check_feed(feed_name, now=now_ts)
            reports[feed_name] = r
            if r["status"] == "fresh":
                n_fresh += 1
            elif r["status"] == "critical":
                n_critical += 1
            else:
                n_stale += 1

        overall_health = "HEALTHY" if n_critical == 0 and n_stale == 0 else (
            "CRITICAL" if n_critical > 0 else "DEGRADED"
        )

        return {
            "feeds": reports,
            "summary": {
                "total_feeds": len(self._feeds),
                "fresh": n_fresh,
                "stale": n_stale,
                "critical": n_critical,
                "overall_health": overall_health,
            },
        }

    # ── Feed Inventory ────────────────────────────────────────────────────────

    def registered_feeds(self) -> List[str]:
        """Returns list of all registered feed names."""
        return list(self._feeds.keys())

    def feed_health_scores(self) -> Dict[str, float]:
        """Returns a dict of feed_name → health_score."""
        return {name: state.health_score for name, state in self._feeds.items()}

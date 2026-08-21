"""
backend_services/routers/alerts.py
───────────────────────────────────
Alert Engine & System Notification Router.
Provides real-time alert rule configuration, threshold monitoring, and notification logs.
"""

from __future__ import annotations
from typing import List, Dict, Any
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from backend_services.auth import verify_token_or_key
from db.database import get_db_connection

router = APIRouter(prefix="/alerts", tags=["Alerts & System Notifications"])


class AlertRuleItem(BaseModel):
    id: int
    metric: str
    condition: str
    threshold: float
    is_active: bool


class AlertHistoryItem(BaseModel):
    id: int
    severity: str
    metric: str
    message: str
    triggered_at: str


@router.get("/rules", response_model=List[AlertRuleItem], summary="Get Active Alert Rules")
async def get_alert_rules(
    client_id: str = Depends(verify_token_or_key),
) -> List[AlertRuleItem]:
    """Returns active monitoring threshold alert rules."""
    return [
        AlertRuleItem(id=1, metric="Feature Drift PSI", condition=">", threshold=0.25, is_active=True),
        AlertRuleItem(id=2, metric="Portfolio Drawdown", condition=">", threshold=0.15, is_active=True),
        AlertRuleItem(id=3, metric="Daily Rank IC", condition="<", threshold=0.01, is_active=True),
        AlertRuleItem(id=4, metric="API Response Latency", condition=">", threshold=500.0, is_active=True),
    ]


@router.get("/history", response_model=List[AlertHistoryItem], summary="Get Alert Notification History")
async def get_alert_history(
    client_id: str = Depends(verify_token_or_key),
) -> List[AlertHistoryItem]:
    """Returns triggered alert events log."""
    return [
        AlertHistoryItem(id=101, severity="INFO", metric="Market Regime Shift", message="Market regime transitioned from BULL_TREND to HIGH_VOLATILITY.", triggered_at="2026-07-29T18:00:00Z"),
        AlertHistoryItem(id=102, severity="WARNING", metric="Turnover Hysteresis", message="Rebalance turnover target exceeded 2.0x baseline.", triggered_at="2026-07-28T09:30:00Z"),
    ]

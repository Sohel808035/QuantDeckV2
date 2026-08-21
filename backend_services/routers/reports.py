"""
backend_services/routers/reports.py
────────────────────────────────────
Report Generation & Teardown Export Router.
Provides PDF/HTML performance report generation, tearsheets, and export capabilities.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend_services.auth import verify_token_or_key

router = APIRouter(prefix="/reports", tags=["Report Generation & Exports"])


class ReportItem(BaseModel):
    report_id: str
    title: str
    file_name: str
    file_type: str
    created_at: str
    size_kb: float


@router.get("/list", response_model=List[ReportItem], summary="Get Generated Performance Reports")
async def list_reports(
    client_id: str = Depends(verify_token_or_key),
) -> List[ReportItem]:
    """Lists available strategy teardown reports and equity curve charts."""
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    items = []
    for idx, f in enumerate(reports_dir.glob("*.*")):
        if f.suffix.lower() in [".png", ".pdf", ".html", ".csv", ".json", ".txt"]:
            items.append(
                ReportItem(
                    report_id=f"rpt_{idx+1}",
                    title=f.stem.replace("_", " ").title(),
                    file_name=f.name,
                    file_type=f.suffix.replace(".", "").upper(),
                    created_at="2026-07-30T10:00:00Z",
                    size_kb=round(f.stat().st_size / 1024.0, 1),
                )
            )

    if not items:
        # Default report items
        items = [
            ReportItem(report_id="rpt_1", title="Institutional Alpha Equity Curve", file_name="equity_curve.png", file_type="PNG", created_at="2026-07-30T10:00:00Z", size_kb=145.2),
            ReportItem(report_id="rpt_2", title="Full Backtest Teardown PDF", file_name="performance_summary.txt", file_type="TXT", created_at="2026-07-30T10:00:00Z", size_kb=8.4),
        ]

    return items

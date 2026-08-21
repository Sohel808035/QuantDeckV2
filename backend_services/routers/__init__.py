"""
Backend APIRouter imports.
"""

from backend_services.routers.health import router as health_router
from backend_services.routers.backtest import router as backtest_router
from backend_services.routers.risk import router as risk_router
from backend_services.routers.monitoring import router as monitoring_router
from backend_services.routers.analyst import router as analyst_router
from backend_services.routers.auth import router as auth_router
from backend_services.routers.stocks import router as stocks_router
from backend_services.routers.predictions import router as predictions_router
from backend_services.routers.portfolio import router as portfolio_router
from backend_services.routers.feature_store import router as feature_store_router
from backend_services.routers.governance import router as governance_router
from backend_services.routers.ai import router as ai_router
from backend_services.routers.alerts import router as alerts_router
from backend_services.routers.reports import router as reports_router

__all__ = [
    "health_router",
    "backtest_router",
    "risk_router",
    "monitoring_router",
    "analyst_router",
    "auth_router",
    "stocks_router",
    "predictions_router",
    "portfolio_router",
    "feature_store_router",
    "governance_router",
    "ai_router",
    "alerts_router",
    "reports_router",
]

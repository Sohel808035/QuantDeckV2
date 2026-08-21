"""
backend_services/app.py
────────────────────────
QuantSphereX Backend Application Factory.
Wires all routers, middleware, exception handlers, and security schemes into a production FastAPI app.
Provides backward compatibility for legacy v1 API endpoints.
"""

from __future__ import annotations
import logging
from typing import Dict
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

try:
    from prometheus_fastapi_instrumentator import Instrumentator as _Instrumentator
    _PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _Instrumentator = None  # type: ignore[assignment,misc]
    _PROMETHEUS_AVAILABLE = False

from backend_services.config import BackendSettings
from backend_services.errors import QuantBackendError, quant_backend_exception_handler, global_unhandled_exception_handler
from backend_services.logger import RequestContextMiddleware
from backend_services.routers import (
    health_router,
    backtest_router,
    risk_router,
    monitoring_router,
    analyst_router,
    auth_router,
    stocks_router,
    predictions_router,
    portfolio_router,
    feature_store_router,
    governance_router,
    ai_router,
    alerts_router,
    reports_router,
)

logger = logging.getLogger(__name__)


def create_app(settings: BackendSettings = BackendSettings()) -> FastAPI:
    """
    FastAPI Application Factory.

    Returns:
        Fully configured FastAPI instance.
    """
    app = FastAPI(
        title=settings.title,
        version=settings.version,
        description=settings.description,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # ── Middleware ───────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(RequestContextMiddleware)

    # ── Exception Handlers ───────────────────────────────────────────────────
    app.add_exception_handler(QuantBackendError, quant_backend_exception_handler)
    app.add_exception_handler(Exception, global_unhandled_exception_handler)

    # ── Prometheus Monitoring ────────────────────────────────────────────────
    if _PROMETHEUS_AVAILABLE and _Instrumentator is not None:
        _Instrumentator().instrument(app).expose(app, endpoint="/metrics")
    else:
        logger.warning("[BackendApp] prometheus_fastapi_instrumentator not installed — /metrics endpoint disabled.")

    # ── v2 APIRouters ────────────────────────────────────────────────────────
    api_v2_prefix = settings.api_prefix
    app.include_router(health_router, prefix=api_v2_prefix)
    app.include_router(backtest_router, prefix=api_v2_prefix)
    app.include_router(risk_router, prefix=api_v2_prefix)
    app.include_router(monitoring_router, prefix=api_v2_prefix)
    app.include_router(analyst_router, prefix=api_v2_prefix)
    app.include_router(auth_router, prefix=api_v2_prefix)
    app.include_router(stocks_router, prefix=api_v2_prefix)
    app.include_router(predictions_router, prefix=api_v2_prefix)
    app.include_router(portfolio_router, prefix=api_v2_prefix)
    app.include_router(feature_store_router, prefix=api_v2_prefix)
    app.include_router(governance_router, prefix=api_v2_prefix)
    app.include_router(ai_router, prefix=api_v2_prefix)
    app.include_router(alerts_router, prefix=api_v2_prefix)
    app.include_router(reports_router, prefix=api_v2_prefix)

    # ── Legacy v1 Routes (Backward Compatibility) ────────────────────────────
    @app.get("/api/v1/health", tags=["Legacy v1 Endpoints"], deprecated=True)
    async def legacy_health_v1():
        """Legacy v1 health endpoint preserved for backward compatibility."""
        return {"status": "ok", "version": "1.0.0", "message": "Deprecation warning: Use /api/v2/health/status"}

    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "name": settings.title,
            "version": settings.version,
            "docs": "/docs",
            "status": "operational",
        }

    logger.info(f"[BackendApp] Created FastAPI app version {settings.version}")
    return app


# Singleton App Instance for ASGI servers (e.g. uvicorn backend_services.app:app)
app = create_app()

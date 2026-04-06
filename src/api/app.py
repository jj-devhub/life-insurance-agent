# git commit: fix(api): replace deprecated on_event with lifespan context manager
# Module: api/app
"""
FastAPI application factory for the Life Insurance Support Assistant.

Configures CORS, error handling, request logging, and registers all
route handlers. Uses modern lifespan context manager for startup/shutdown.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.models import ErrorResponse
from src.api.routes.chat import router as chat_router
from src.api.routes.health import router as health_router
from src.api.routes.knowledge import kb_router, memory_router
from src.config import get_settings

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Lifespan context manager (replaces deprecated on_event)
# --------------------------------------------------------------------------- #


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle for the FastAPI application."""
    settings = get_settings()

    # ---- Startup ---- #
    logging.basicConfig(
        level=getattr(logging, settings.log_level.value),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    logger.info("🚀 Starting %s", settings.app_name)
    logger.info("   LLM Provider: %s (%s)", settings.llm_provider.value, settings.active_model)
    logger.info("   Mem0 Enabled: %s", settings.mem0_enabled)
    logger.info("   KB Path: %s", settings.kb_path)

    yield  # Application runs here

    # ---- Shutdown ---- #
    logger.info("Shutting down %s", settings.app_name)


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application instance.

    Returns:
        Configured FastAPI app with all routes and middleware.
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description=(
            "AI-powered Life Insurance Support Assistant API. "
            "Uses LangGraph multi-agent workflows, Mem0 persistent memory, "
            "and a configurable YAML knowledge base to answer life insurance queries."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ---- CORS Middleware ---- #
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Restrict in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---- Request Logging Middleware ---- #
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time
        logger.info(
            "%s %s → %d (%.2fs)",
            request.method,
            request.url.path,
            response.status_code,
            duration,
        )
        return response

    # ---- Global Exception Handler ---- #
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error("Unhandled exception: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="Internal Server Error",
                detail=str(exc) if settings.debug else "An unexpected error occurred.",
                status_code=500,
            ).model_dump(),
        )

    # ---- Register Routes ---- #
    app.include_router(chat_router)
    app.include_router(kb_router)
    app.include_router(memory_router)
    app.include_router(health_router)

    # ---- Root endpoint ---- #
    @app.get("/", tags=["Root"])
    async def root():
        return {
            "app": settings.app_name,
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/api/v1/health",
        }

    return app


# Module-level app instance (used by uvicorn)
app = create_app()


def start_server():
    """Entry point for the lia-server command."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "src.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    start_server()

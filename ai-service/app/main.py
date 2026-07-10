"""
FastAPI application entry point.

Configures middleware, logging, startup model loading, and route registration.
Models are loaded once at startup and kept in memory for the process lifetime.
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import APP_DESCRIPTION, APP_TITLE, APP_VERSION, HOST, LOG_LEVEL, PORT
from routes.verify import router as verify_router
from services.model_loader import model_loader

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application lifespan — load models once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan hook.

    Loads all TensorFlow models into memory before accepting requests.
    Models remain cached for the entire process lifetime.
    """
    logger.info("Starting AI Image Authenticity Service v%s", APP_VERSION)
    try:
        model_loader.load_all()
        logger.info("Startup complete — service ready to accept requests.")
    except Exception:
        logger.exception("Failed to load models during startup.")
        raise

    yield

    logger.info("Shutting down AI Image Authenticity Service.")


# ---------------------------------------------------------------------------
# FastAPI application instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow Express/Node.js backend to call this service cross-origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request ID middleware — attaches a unique ID to every request for tracing
# ---------------------------------------------------------------------------

@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Assign a request ID and log request timing for every HTTP call."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id

    start = time.perf_counter()
    logger.info(
        "Incoming request request_id=%s method=%s path=%s",
        request_id,
        request.method,
        request.url.path,
    )

    response = await call_next(request)
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    response.headers["X-Request-ID"] = request_id
    logger.info(
        "Request completed request_id=%s status=%d elapsed_ms=%d",
        request_id,
        response.status_code,
        elapsed_ms,
    )
    return response


# ---------------------------------------------------------------------------
# Global exception handler — ensures all errors return clean JSON (no HTML)
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch unhandled exceptions and return a JSON 500 response."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.exception(
        "Unhandled exception request_id=%s path=%s error=%s",
        request_id,
        request.url.path,
        str(exc),
    )
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "detail": "Internal server error.",
            "request_id": request_id,
        },
    )


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------

app.include_router(verify_router)


# ---------------------------------------------------------------------------
# Direct execution entry point (development)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=False)

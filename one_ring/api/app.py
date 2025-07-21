"""
FastAPI application factory for the One Ring API.

This module creates and configures the FastAPI application with all routes,
middleware, and event handlers.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from one_ring.api.v1.endpoints import models as models_router
from one_ring.api.v1.endpoints import documents as documents_router
from one_ring.api.v1.endpoints import training as training_router
from one_ring.core.config import settings
from one_ring.core.hardware import hardware_manager
from one_ring.core.logger import setup_logging
from one_ring.services.model_service import model_service
from one_ring.utils.cleanup import register_cleanup_handler


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app."""
    # Startup
    logger.info("Starting One Ring API...")
    
    # Log hardware information
    hardware_info = hardware_manager.get_hardware_info()
    logger.info(f"Hardware: {hardware_info}")
    
    # Initialize model service
    try:
        # Load default model if specified in settings
        if settings.DEFAULT_MODEL:
            logger.info(f"Loading default model: {settings.DEFAULT_MODEL}")
            model_service.load_model(
                model_name=settings.DEFAULT_MODEL,
                model_type=settings.DEFAULT_MODEL_TYPE,
            )
    except Exception as e:
        logger.error(f"Failed to load default model: {e}")
    
    yield  # App is running
    
    # Shutdown
    logger.info("Shutting down One Ring API...")
    
    # Clean up resources
    model_service.cleanup()
    
    logger.info("One Ring API shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.
    
    Returns:
        A configured FastAPI application instance.
    """
    # Configure logging
    setup_logging()
    
    # Create FastAPI app with lifespan
    app = FastAPI(
        title="One Ring API",
        description="API for the One Ring AI Fine-Tuning Platform",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add exception handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        """Handle HTTP exceptions."""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail.get("error", "An error occurred"),
                "details": exc.detail.get("details", {}),
            },
        )
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle all other exceptions."""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal server error",
                "details": {"message": str(exc)},
            },
        )
    
    # Add middleware for request/response logging
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log all incoming requests and outgoing responses."""
        logger.info(f"Request: {request.method} {request.url}")
        
        try:
            response = await call_next(request)
            logger.info(f"Response: {request.method} {request.url} - {response.status_code}")
            return response
        except Exception as e:
            logger.error(f"Error processing request: {e}", exc_info=True)
            raise
    
    # Include API routers
    app.include_router(models_router.router, prefix="/api/v1")
    app.include_router(documents_router.router, prefix="/api/v1")
    app.include_router(training_router.router, prefix="/api/v1")
    
    # Health check endpoint
    @app.get("/health", tags=["health"])
    async def health_check() -> Dict[str, str]:
        """Health check endpoint."""
        return {"status": "ok"}
    
    # Root endpoint
    @app.get("/", tags=["root"])
    async def root() -> Dict[str, Any]:
        """Root endpoint with API information."""
        return {
            "name": "One Ring API",
            "version": "0.1.0",
            "description": "API for the One Ring AI Fine-Tuning Platform",
            "documentation": "/docs",
            "health_check": "/health",
        }
    
    return app


# Create the app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    # Run the FastAPI app with Uvicorn
    uvicorn.run(
        "one_ring.api.app:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level="info",
    )

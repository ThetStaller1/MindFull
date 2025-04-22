from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
import logging
import time
from typing import Dict, Any

# Import routers
from api.healthdata import router as healthdata_router
from api.auth import router as auth_router
from api.users import router as users_router
from api.mental_health import router as mental_health_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="MindFull Health API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (update for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    logger.info(f"Request to {request.url.path} took {process_time} seconds")
    return response

# Include routers
app.include_router(auth_router, tags=["Authentication"], prefix="")
app.include_router(healthdata_router, tags=["Health Data"], prefix="/api")
app.include_router(users_router, tags=["Users"], prefix="/api")
app.include_router(mental_health_router, tags=["Mental Health"], prefix="/api")

@app.get("/health", tags=["Health Check"])
async def health_check() -> Dict[str, Any]:
    """Check if the API is running."""
    return {
        "status": "ok",
        "version": "1.0.0"
    }

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Uncaught exception: {exc}", exc_info=True)
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"Internal server error: {str(exc)}"
    ) 
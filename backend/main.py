from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from config import settings
from api.chat import router as chat_router

# Setup FastAPI
app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    description="Production-ready RAG API for conference Q&A"
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # OK per MVP/demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router, prefix="/api", tags=["chat"])


@app.get("/")
def root():
    return {
        "status": "running",
        "version": settings.app_version,
        "docs": "/docs"
    }


@app.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "embeddings_file": settings.embeddings_file,
        "llm_model": settings.llm_model,
    }
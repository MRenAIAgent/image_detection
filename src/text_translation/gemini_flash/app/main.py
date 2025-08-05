"""
Gemini Flash Translation API - Main FastAPI Application
"""
import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from .config import settings
from .gemini_client import translation_client

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = structlog.get_logger(__name__)


# Pydantic models
class TranslationRequest(BaseModel):
    """Single translation request"""
    text: str = Field(..., min_length=1, max_length=settings.max_text_length)
    source_language: str = Field(..., min_length=1)
    target_language: str = Field(..., min_length=1)
    use_cache: bool = Field(default=True)
    
    @validator('text')
    def text_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Text cannot be empty')
        return v.strip()


class BatchTranslationRequest(BaseModel):
    """Batch translation request"""
    texts: List[str] = Field(..., min_items=1, max_items=settings.max_batch_size)
    source_language: str = Field(..., min_length=1)
    target_language: str = Field(..., min_length=1)
    use_cache: bool = Field(default=True)
    
    @validator('texts')
    def texts_not_empty(cls, v):
        if not v:
            raise ValueError('Texts list cannot be empty')
        for i, text in enumerate(v):
            if not text or not text.strip():
                raise ValueError(f'Text at index {i} cannot be empty')
            if len(text) > settings.max_text_length:
                raise ValueError(f'Text at index {i} exceeds maximum length of {settings.max_text_length}')
        return [text.strip() for text in v]


class TranslationResponse(BaseModel):
    """Translation response"""
    translated_text: str
    source_language: str
    target_language: str
    original_text: str
    inference_time: float
    model: str
    cached: bool = False


class BatchTranslationResponse(BaseModel):
    """Batch translation response"""
    translations: List[Dict[str, Any]]
    total_texts: int
    successful_translations: int
    failed_translations: int
    total_inference_time: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: float
    version: str
    model: str
    uptime: float


class StatsResponse(BaseModel):
    """Statistics response"""
    total_requests: int
    total_characters: int
    total_inference_time: float
    avg_inference_time: float
    avg_characters_per_request: float
    errors: int
    cache_hits: int
    cache_hit_rate: float
    cache_size: int


# Application lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("Starting Gemini Flash Translation API...")
    
    try:
        await translation_client.initialize()
        logger.info("✅ Gemini client initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Gemini client: {e}")
        raise
    
    # Store startup time
    app.state.startup_time = time.time()
    
    yield
    
    # Shutdown
    logger.info("Shutting down Gemini Flash Translation API...")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description=settings.description,
    version=settings.app_version,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=settings.cors_methods,
    allow_headers=settings.cors_headers,
)


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    uptime = time.time() - getattr(app.state, 'startup_time', time.time())
    
    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        version=settings.app_version,
        model=settings.gemini_model,
        uptime=uptime
    )


@app.post("/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    """Translate a single text"""
    try:
        logger.info(
            "Translation request",
            source_lang=request.source_language,
            target_lang=request.target_language,
            text_length=len(request.text),
            use_cache=request.use_cache
        )
        
        result = await translation_client.translate(
            text=request.text,
            source_language=request.source_language,
            target_language=request.target_language,
            use_cache=request.use_cache
        )
        
        logger.info(
            "Translation completed",
            inference_time=result["inference_time"],
            cached=result.get("cached", False)
        )
        
        return TranslationResponse(**result)
        
    except ValueError as e:
        logger.warning(f"Invalid request: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Translation failed"
        )


@app.post("/translate/batch", response_model=BatchTranslationResponse)
async def translate_batch(request: BatchTranslationRequest):
    """Translate multiple texts in batch"""
    try:
        logger.info(
            "Batch translation request",
            source_lang=request.source_language,
            target_lang=request.target_language,
            batch_size=len(request.texts),
            use_cache=request.use_cache
        )
        
        start_time = time.time()
        results = await translation_client.translate_batch(
            texts=request.texts,
            source_language=request.source_language,
            target_language=request.target_language,
            use_cache=request.use_cache
        )
        total_time = time.time() - start_time
        
        # Count successful and failed translations
        successful = sum(1 for r in results if "error" not in r)
        failed = len(results) - successful
        
        logger.info(
            "Batch translation completed",
            total_texts=len(request.texts),
            successful=successful,
            failed=failed,
            total_time=total_time
        )
        
        return BatchTranslationResponse(
            translations=results,
            total_texts=len(request.texts),
            successful_translations=successful,
            failed_translations=failed,
            total_inference_time=total_time
        )
        
    except ValueError as e:
        logger.warning(f"Invalid batch request: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Batch translation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch translation failed"
        )


@app.get("/languages", response_model=Dict[str, Any])
async def get_supported_languages():
    """Get list of supported languages"""
    return {
        "supported_languages": settings.supported_languages,
        "total_languages": len(settings.supported_languages),
        "language_mappings": settings.language_mappings,
        "note": "You can use either language codes (e.g., 'en', 'es') or full names (e.g., 'english', 'spanish')"
    }


@app.get("/model/info", response_model=Dict[str, Any])
async def get_model_info():
    """Get model information"""
    return translation_client.get_model_info()


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get translation statistics"""
    stats = translation_client.get_stats()
    return StatsResponse(**stats)


@app.post("/stats/reset")
async def reset_stats():
    """Reset translation statistics"""
    translation_client.reset_stats()
    return {"message": "Statistics reset successfully"}


@app.post("/cache/clear")
async def clear_cache():
    """Clear translation cache"""
    translation_client.clear_cache()
    return {"message": "Cache cleared successfully"}


@app.get("/cache/info")
async def get_cache_info():
    """Get cache information"""
    stats = translation_client.get_stats()
    return {
        "cache_size": stats["cache_size"],
        "cache_hits": stats["cache_hits"],
        "cache_hit_rate": stats["cache_hit_rate"],
        "max_cache_size": translation_client._max_cache_size
    }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": time.time()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": time.time()
        }
    )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
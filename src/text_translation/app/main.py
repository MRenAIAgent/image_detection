"""
Main FastAPI application for Text Translation API using NLLB-200
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from .config import settings, validate_language_pair, validate_text_length, get_nllb_language_code
from .translation_client import translation_client

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format=settings.log_format
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    
    # Startup
    logger.info("Starting Text Translation API...")
    
    # Initialize translation client
    if not await translation_client.initialize():
        logger.error("Failed to initialize translation client")
        raise RuntimeError("Failed to initialize translation client")
    
    logger.info("Text Translation API started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Text Translation API...")
    logger.info("Text Translation API shutdown complete")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="High-performance text translation API using NLLB-200 with quantization support",
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


# Request/Response models
class TranslationRequest(BaseModel):
    """Single translation request."""
    text: str = Field(..., description="Text to translate", max_length=settings.max_text_length)
    source_language: str = Field(..., description="Source language (e.g., 'english', 'en', 'eng_Latn')")
    target_language: str = Field(..., description="Target language (e.g., 'spanish', 'es', 'spa_Latn')")


class BatchTranslationRequest(BaseModel):
    """Batch translation request."""
    texts: List[str] = Field(..., description="List of texts to translate", max_items=settings.max_batch_size)
    source_language: str = Field(..., description="Source language")
    target_language: str = Field(..., description="Target language")


class TranslationResponse(BaseModel):
    """Translation response."""
    translated_text: str
    source_language: str
    target_language: str
    original_text: str
    inference_time: float
    confidence: Optional[float] = None


class BatchTranslationResponse(BaseModel):
    """Batch translation response."""
    translations: List[TranslationResponse]
    total_time: float
    batch_size: int
    average_time_per_translation: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_healthy: bool
    model_loaded: bool
    timestamp: float


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_name: str
    device: str
    quantization_enabled: bool
    quantization_type: Optional[str]
    model_loaded: bool
    supported_languages: int
    max_length: int
    cache_enabled: bool
    stats: Dict[str, Any]


class SupportedLanguagesResponse(BaseModel):
    """Supported languages response."""
    languages: Dict[str, str]
    total_count: int


# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Text Translation API",
        "version": settings.app_version,
        "model": settings.model_name,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        model_healthy = await translation_client.health_check()
        model_loaded = translation_client.model_loaded
        
        status = "healthy" if model_healthy and model_loaded else "unhealthy"
        
        return HealthResponse(
            status=status,
            model_healthy=model_healthy,
            model_loaded=model_loaded,
            timestamp=time.time()
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthResponse(
            status="unhealthy",
            model_healthy=False,
            model_loaded=False,
            timestamp=time.time()
        )


@app.post("/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    """
    Translate text from source to target language.
    
    Args:
        request: Translation request with text and language pair
        
    Returns:
        Translation result with timing information
    """
    try:
        # Validate text length
        if not validate_text_length(request.text, settings.max_text_length):
            raise HTTPException(
                status_code=400,
                detail=f"Text too long. Maximum length: {settings.max_text_length} characters"
            )
        
        # Validate and convert language codes
        try:
            source_code, target_code = validate_language_pair(
                request.source_language, 
                request.target_language
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Perform translation
        result = await translation_client.translate(
            request.text, 
            source_code, 
            target_code
        )
        
        return TranslationResponse(
            translated_text=result.translated_text,
            source_language=request.source_language,
            target_language=request.target_language,
            original_text=request.text,
            inference_time=result.inference_time,
            confidence=result.confidence
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Translation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/translate/batch", response_model=BatchTranslationResponse)
async def translate_batch(request: BatchTranslationRequest):
    """
    Translate multiple texts in batch.
    
    Args:
        request: Batch translation request
        
    Returns:
        Batch translation results
    """
    try:
        if not request.texts:
            raise HTTPException(status_code=400, detail="No texts provided")
        
        if len(request.texts) > settings.max_batch_size:
            raise HTTPException(
                status_code=400,
                detail=f"Too many texts. Maximum batch size: {settings.max_batch_size}"
            )
        
        # Validate text lengths
        for i, text in enumerate(request.texts):
            if not validate_text_length(text, settings.max_text_length):
                raise HTTPException(
                    status_code=400,
                    detail=f"Text {i+1} too long. Maximum length: {settings.max_text_length} characters"
                )
        
        # Validate and convert language codes
        try:
            source_code, target_code = validate_language_pair(
                request.source_language, 
                request.target_language
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        start_time = time.time()
        
        # Perform batch translation
        results = await translation_client.translate_batch(
            request.texts, 
            source_code, 
            target_code
        )
        
        total_time = time.time() - start_time
        
        # Convert results
        translations = []
        for i, result in enumerate(results):
            translations.append(TranslationResponse(
                translated_text=result.translated_text,
                source_language=request.source_language,
                target_language=request.target_language,
                original_text=request.texts[i],
                inference_time=result.inference_time,
                confidence=result.confidence
            ))
        
        return BatchTranslationResponse(
            translations=translations,
            total_time=total_time,
            batch_size=len(request.texts),
            average_time_per_translation=total_time / len(request.texts)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch translation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/languages", response_model=SupportedLanguagesResponse)
async def get_supported_languages():
    """Get list of supported languages."""
    try:
        from .config import NLLB_LANGUAGE_CODES
        
        return SupportedLanguagesResponse(
            languages=NLLB_LANGUAGE_CODES,
            total_count=len(NLLB_LANGUAGE_CODES)
        )
        
    except Exception as e:
        logger.error(f"Error getting supported languages: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get model information and statistics."""
    try:
        info = translation_client.get_model_info()
        
        return ModelInfoResponse(
            model_name=info["model_name"],
            device=info["device"],
            quantization_enabled=info["quantization_enabled"],
            quantization_type=info["quantization_type"],
            model_loaded=info["model_loaded"],
            supported_languages=info["supported_languages"],
            max_length=info["max_length"],
            cache_enabled=info["cache_enabled"],
            stats=info["stats"]
        )
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_statistics():
    """Get translation statistics."""
    try:
        return translation_client.get_stats()
        
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stats/reset")
async def reset_statistics():
    """Reset translation statistics."""
    try:
        translation_client.clear_stats()
        return {"message": "Statistics reset successfully"}
        
    except Exception as e:
        logger.error(f"Error resetting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cache/clear")
async def clear_cache():
    """Clear translation cache."""
    try:
        translation_client.clear_cache()
        return {"message": "Cache cleared successfully"}
        
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "timestamp": time.time()}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "timestamp": time.time()}
    )


# Main function for running the app
def main():
    """Run the FastAPI application."""
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level=settings.log_level.lower() if hasattr(settings.log_level, 'lower') else "info",
        reload=settings.debug
    )


if __name__ == "__main__":
    main()
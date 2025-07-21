"""
Main FastAPI application for Image Recognition API
"""

# Fix for PyTorch 2.6 weights_only issue with YOLO models
import os
os.environ['PYTORCH_DISABLE_WEIGHTS_ONLY_LOAD'] = '1'

# Patch torch.load to use weights_only=False for compatibility
import torch
_original_load = torch.load

def patched_load(*args, **kwargs):
    """Patched torch.load that defaults to weights_only=False for YOLO compatibility."""
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)

torch.load = patched_load

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from .config import settings, validate_file_extension, validate_file_size
from .triton_client import triton_client
from .batch_processor import BatchProcessor

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format=settings.log_format
)
logger = logging.getLogger(__name__)

# Global batch processor
batch_processor: Optional[BatchProcessor] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global batch_processor
    
    # Startup
    logger.info("Starting Image Recognition API...")
    
    # Initialize Triton client
    if not await triton_client.initialize():
        logger.error("Failed to initialize Triton client")
        raise RuntimeError("Failed to initialize Triton client")
    
    # Initialize batch processor
    batch_processor = BatchProcessor(triton_client)
    await batch_processor.start()
    
    logger.info("Image Recognition API started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Image Recognition API...")
    
    if batch_processor:
        await batch_processor.stop()
    
    triton_client.close()
    logger.info("Image Recognition API shutdown complete")

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="High-performance image recognition API using YOLOv8n with NVIDIA Triton Inference Server",
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

# Response models
class Detection(BaseModel):
    """Object detection result."""
    class_id: int
    class_name: str
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2]

class SingleDetectionResponse(BaseModel):
    """Response for single image detection."""
    detections: List[Detection]
    inference_time: float
    filename: str

class BatchDetectionResponse(BaseModel):
    """Response for batch image detection."""
    results: List[SingleDetectionResponse]
    total_time: float
    batch_size: int

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    triton_healthy: bool
    model_ready: bool
    timestamp: float

class StatsResponse(BaseModel):
    """Statistics response."""
    batch_processor: Dict[str, Any]
    model_info: Dict[str, Any]

# Dependency for file validation
async def validate_uploaded_file(file: UploadFile = File(...)) -> UploadFile:
    """Validate uploaded file."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    if not validate_file_extension(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file extension. Allowed: {', '.join(['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp'])}"
        )
    
    # Read file to check size
    content = await file.read()
    if not validate_file_size(len(content)):
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {settings.max_file_size / (1024*1024):.1f}MB"
        )
    
    # Reset file position
    await file.seek(0)
    return file

# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Image Recognition API",
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    triton_healthy = await triton_client.health_check()
    model_ready = triton_healthy  # If triton is healthy, model should be ready
    
    status = "healthy" if triton_healthy and model_ready else "unhealthy"
    
    return HealthResponse(
        status=status,
        triton_healthy=triton_healthy,
        model_ready=model_ready,
        timestamp=time.time()
    )

@app.post("/detect", response_model=SingleDetectionResponse)
async def detect_single_image(
    file: UploadFile = Depends(validate_uploaded_file)
):
    """
    Detect objects in a single image.
    
    Args:
        file: Image file (JPG, PNG, etc.)
        
    Returns:
        Detection results with bounding boxes and confidence scores
    """
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Process through batch processor for consistency
        result = await batch_processor.add_request(image_bytes, file.filename)
        
        if result.error:
            raise HTTPException(status_code=500, detail=result.error)
        
        # Convert to response format
        detections = [
            Detection(
                class_id=det["class_id"],
                class_name=det["class_name"],
                confidence=det["confidence"],
                bbox=det["bbox"]
            )
            for det in result.detections
        ]
        
        return SingleDetectionResponse(
            detections=detections,
            inference_time=result.inference_time,
            filename=file.filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing single image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/batch", response_model=BatchDetectionResponse)
async def detect_batch_images(
    files: List[UploadFile] = File(...)
):
    """
    Detect objects in multiple images.
    
    Args:
        files: List of image files
        
    Returns:
        Detection results for all images
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    if len(files) > settings.max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Maximum batch size: {settings.max_batch_size}"
        )
    
    start_time = time.time()
    results = []
    
    try:
        # Process all files
        tasks = []
        for file in files:
            # Validate file
            if not validate_file_extension(file.filename):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file extension for {file.filename}"
                )
            
            # Read image bytes
            image_bytes = await file.read()
            
            if not validate_file_size(len(image_bytes)):
                raise HTTPException(
                    status_code=413,
                    detail=f"File {file.filename} too large"
                )
            
            # Add to batch processor
            task = batch_processor.add_request(image_bytes, file.filename)
            tasks.append(task)
        
        # Wait for all results
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error(f"Error processing file {files[i].filename}: {result}")
                results.append(SingleDetectionResponse(
                    detections=[],
                    inference_time=0.0,
                    filename=files[i].filename
                ))
            else:
                detections = [
                    Detection(
                        class_id=det["class_id"],
                        class_name=det["class_name"],
                        confidence=det["confidence"],
                        bbox=det["bbox"]
                    )
                    for det in result.detections
                ]
                
                results.append(SingleDetectionResponse(
                    detections=detections,
                    inference_time=result.inference_time,
                    filename=result.filename
                ))
        
        total_time = time.time() - start_time
        
        return BatchDetectionResponse(
            results=results,
            total_time=total_time,
            batch_size=len(files)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats", response_model=StatsResponse)
async def get_statistics():
    """Get API statistics."""
    try:
        batch_stats = batch_processor.get_stats() if batch_processor else {}
        model_info = await triton_client.get_model_info()
        
        return StatsResponse(
            batch_processor=batch_stats,
            model_info=model_info
        )
        
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stats/reset")
async def reset_statistics():
    """Reset API statistics."""
    try:
        if batch_processor:
            batch_processor.clear_stats()
        
        return {"message": "Statistics reset successfully"}
        
    except Exception as e:
        logger.error(f"Error resetting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info")
async def get_model_info():
    """Get model information."""
    try:
        return await triton_client.get_model_info()
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
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
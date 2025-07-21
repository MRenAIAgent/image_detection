"""
Local development version of the Image Recognition API
Runs YOLOv8n directly without Triton for Mac testing
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
import io
import logging
import time
from typing import List, Dict, Any
import numpy as np
from PIL import Image
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

# Import YOLOv8 directly
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# COCO class names
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

class LocalYOLOClient:
    """Local YOLO client for direct inference without Triton."""
    
    def __init__(self):
        self.model = None
        self.confidence_threshold = 0.5
        self.model_loaded = False
        
    async def initialize(self):
        """Initialize YOLO model."""
        if YOLO is None:
            logger.error("Ultralytics YOLO not available")
            return False
            
        try:
            # Try to load ONNX model first, fallback to PyTorch
            model_paths = [
                "models/model_repository/yolov8n/1/model.onnx",
                "yolov8n.onnx", 
                "yolov8n.pt"
            ]
            
            for model_path in model_paths:
                try:
                    logger.info(f"Trying to load model from: {model_path}")
                    self.model = YOLO(model_path)
                    logger.info(f"Successfully loaded model from: {model_path}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load {model_path}: {e}")
                    continue
            
            if self.model is None:
                # Download YOLOv8n if no local model found
                logger.info("No local model found, downloading YOLOv8n...")
                self.model = YOLO('yolov8n.pt')
                
            self.model_loaded = True
            logger.info("YOLO model initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize YOLO model: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Check if model is healthy."""
        return self.model_loaded and self.model is not None
    
    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """Preprocess image for YOLO inference."""
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(image)
        
        return img_array
    
    def postprocess_detections(self, results, filename: str = None) -> Dict[str, Any]:
        """Convert YOLO results to API format."""
        detections = []
        
        if results and len(results) > 0:
            result = results[0]  # First result
            
            if result.boxes is not None:
                boxes = result.boxes
                
                for i in range(len(boxes)):
                    # Get box coordinates
                    box = boxes.xyxy[i].cpu().numpy()
                    confidence = float(boxes.conf[i].cpu().numpy())
                    class_id = int(boxes.cls[i].cpu().numpy())
                    
                    # Filter by confidence threshold
                    if confidence >= self.confidence_threshold:
                        detection = {
                            "class_id": class_id,
                            "class_name": COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}",
                            "confidence": round(confidence, 3),
                            "bbox": [
                                round(float(box[0]), 1),  # x1
                                round(float(box[1]), 1),  # y1
                                round(float(box[2]), 1),  # x2
                                round(float(box[3]), 1)   # y2
                            ]
                        }
                        detections.append(detection)
        
        return {
            "detections": detections,
            "filename": filename or "unknown",
            "num_detections": len(detections)
        }
    
    async def infer(self, images: List[bytes], filenames: List[str] = None) -> List[Dict[str, Any]]:
        """Run inference on batch of images."""
        if not self.model_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        results = []
        
        for i, image_bytes in enumerate(images):
            start_time = time.time()
            
            try:
                # Preprocess image
                img_array = self.preprocess_image(image_bytes)
                
                # Run inference (let YOLO detect everything, filter in postprocessing)
                yolo_results = self.model(img_array, conf=0.1, verbose=False)
                
                # Postprocess results
                filename = filenames[i] if filenames and i < len(filenames) else f"image_{i}"
                result = self.postprocess_detections(yolo_results, filename)
                
                # Add timing info
                inference_time = time.time() - start_time
                result["inference_time"] = round(inference_time, 3)
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Inference failed for image {i}: {e}")
                results.append({
                    "error": str(e),
                    "filename": filenames[i] if filenames and i < len(filenames) else f"image_{i}",
                    "detections": []
                })
        
        return results

# Initialize FastAPI app
app = FastAPI(
    title="Image Recognition API (Local)",
    description="Local development version of YOLOv8n object detection API",
    version="1.0.0-local"
)

# Initialize YOLO client
yolo_client = LocalYOLOClient()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting Image Recognition API (Local Mode)")
    
    success = await yolo_client.initialize()
    if not success:
        logger.error("Failed to initialize YOLO model")
        raise RuntimeError("Model initialization failed")
    
    logger.info("🚀 API ready for local testing!")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Image Recognition API (Local Development)",
        "version": "1.0.0-local",
        "model": "YOLOv8n",
        "mode": "local",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    model_healthy = await yolo_client.health_check()
    
    return {
        "status": "healthy" if model_healthy else "unhealthy",
        "model_loaded": model_healthy,
        "mode": "local",
        "timestamp": time.time()
    }

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    """Detect objects in a single image."""
    # Validate file type
    if not file:
        raise HTTPException(status_code=400, detail="File is required")
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Run inference
        results = await yolo_client.infer([image_bytes], [file.filename])
        
        return results[0]  # Return single result
        
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/detect/batch")
async def detect_objects_batch(files: List[UploadFile] = File(...)):
    """Detect objects in multiple images."""
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    
    # Validate all files are images
    for file in files:
        if not file:
            raise HTTPException(status_code=400, detail="File is required")
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail=f"File {file.filename} must be an image")
    
    try:
        # Read all image bytes
        images_bytes = []
        filenames = []
        
        for file in files:
            image_bytes = await file.read()
            images_bytes.append(image_bytes)
            filenames.append(file.filename)
        
        # Run batch inference
        results = await yolo_client.infer(images_bytes, filenames)
        
        return {
            "results": results,
            "total_images": len(results),
            "batch_size": len(files)
        }
        
    except Exception as e:
        logger.error(f"Batch detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch detection failed: {str(e)}")

@app.get("/model/info")
async def get_model_info():
    """Get model information."""
    return {
        "model_name": "YOLOv8n",
        "model_type": "object_detection",
        "classes": len(COCO_CLASSES),
        "class_names": COCO_CLASSES,
        "input_size": [640, 640],
        "confidence_threshold": yolo_client.confidence_threshold,
        "mode": "local"
    }

@app.get("/stats")
async def get_stats():
    """Get API statistics."""
    return {
        "model_loaded": yolo_client.model_loaded,
        "mode": "local",
        "confidence_threshold": yolo_client.confidence_threshold,
        "supported_formats": ["jpg", "jpeg", "png", "bmp", "tiff"],
        "max_batch_size": 10
    }

if __name__ == "__main__":
    # Run the API locally
    logger.info("Starting local Image Recognition API...")
    import os
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(
        "local_main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    ) 
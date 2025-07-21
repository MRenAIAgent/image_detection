"""
Configuration management for the Image Recognition API
"""

import os
from typing import List, Optional
from pydantic import BaseModel, Field

class Settings(BaseModel):
    """Application settings with environment variable support."""
    
    # API Configuration
    app_name: str = Field(default="Image Recognition API", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8080, env="PORT")
    workers: int = Field(default=1, env="WORKERS")
    
    # Triton Configuration
    triton_url: str = Field(default="localhost:8001", env="TRITON_URL")
    triton_model_name: str = Field(default="yolov8n", env="TRITON_MODEL_NAME")
    triton_model_version: str = Field(default="1", env="TRITON_MODEL_VERSION")
    triton_timeout: float = Field(default=30.0, env="TRITON_TIMEOUT")
    triton_retry_attempts: int = Field(default=3, env="TRITON_RETRY_ATTEMPTS")
    triton_retry_delay: float = Field(default=1.0, env="TRITON_RETRY_DELAY")
    
    # Model Configuration
    model_input_shape: List[int] = Field(default=[640, 640], env="MODEL_INPUT_SHAPE")
    confidence_threshold: float = Field(default=0.5, env="CONFIDENCE_THRESHOLD")
    nms_threshold: float = Field(default=0.4, env="NMS_THRESHOLD")
    max_detections: int = Field(default=100, env="MAX_DETECTIONS")
    
    # Batch Processing Configuration
    max_batch_size: int = Field(default=16, env="MAX_BATCH_SIZE")
    batch_timeout: float = Field(default=0.1, env="BATCH_TIMEOUT")  # seconds
    max_queue_size: int = Field(default=100, env="MAX_QUEUE_SIZE")
    
    # File Upload Configuration
    max_file_size: int = Field(default=10 * 1024 * 1024, env="MAX_FILE_SIZE")  # 10MB
    allowed_extensions: List[str] = Field(
        default=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
        env="ALLOWED_EXTENSIONS"
    )
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    
    # Monitoring Configuration
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=8090, env="METRICS_PORT")
    
    # CORS Configuration
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    cors_methods: List[str] = Field(default=["GET", "POST"], env="CORS_METHODS")
    cors_headers: List[str] = Field(default=["*"], env="CORS_HEADERS")
    
    # Health Check Configuration
    health_check_interval: float = Field(default=30.0, env="HEALTH_CHECK_INTERVAL")
    
    # Performance Configuration
    enable_gpu: bool = Field(default=True, env="ENABLE_GPU")
    gpu_memory_fraction: float = Field(default=0.8, env="GPU_MEMORY_FRACTION")
    
    # Production Configuration
    environment: str = Field(default="development", env="ENVIRONMENT")
    request_timeout: float = Field(default=300.0, env="REQUEST_TIMEOUT")
    max_concurrent_requests: int = Field(default=100, env="MAX_CONCURRENT_REQUESTS")
    
    # Rate Limiting
    enable_rate_limiting: bool = Field(default=True, env="ENABLE_RATE_LIMITING")
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")  # per minute
    rate_limit_window: int = Field(default=60, env="RATE_LIMIT_WINDOW")  # seconds
    
    # Caching Configuration
    enable_caching: bool = Field(default=False, env="ENABLE_CACHING")
    cache_ttl: int = Field(default=300, env="CACHE_TTL")  # seconds
    
    # Security Configuration
    api_key_required: bool = Field(default=False, env="API_KEY_REQUIRED")
    api_key_header: str = Field(default="X-API-Key", env="API_KEY_HEADER")
    allowed_api_keys: List[str] = Field(default=[], env="ALLOWED_API_KEYS")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# Global settings instance
settings = Settings()

# COCO class names for YOLOv8n
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

def get_class_name(class_id: int) -> str:
    """Get class name from class ID."""
    if 0 <= class_id < len(COCO_CLASSES):
        return COCO_CLASSES[class_id]
    return f"unknown_{class_id}"

def validate_file_extension(filename: str) -> bool:
    """Validate if file extension is allowed."""
    if not filename:
        return False
    
    extension = filename.lower().split('.')[-1]
    return extension in [ext.lower() for ext in settings.allowed_extensions]

def validate_file_size(file_size: int) -> bool:
    """Validate if file size is within limits."""
    return file_size <= settings.max_file_size

def is_production() -> bool:
    """Check if running in production environment."""
    return settings.environment.lower() in ["production", "prod"]

def get_cors_origins() -> List[str]:
    """Get CORS origins based on environment."""
    if is_production():
        # In production, restrict CORS origins
        return [origin for origin in settings.cors_origins if origin != "*"]
    return settings.cors_origins 
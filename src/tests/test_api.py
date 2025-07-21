"""
Comprehensive tests for the Image Recognition API
"""

import pytest
import asyncio
import io
from typing import List
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np

# Mock the triton client import to avoid import errors during testing
import sys
from unittest.mock import MagicMock

# Mock tritonclient
mock_tritonclient = MagicMock()
sys.modules['tritonclient'] = mock_tritonclient
sys.modules['tritonclient.grpc'] = mock_tritonclient.grpc
sys.modules['tritonclient.utils'] = mock_tritonclient.utils

from app.main import app
from app.config import settings
from app.batch_processor import BatchProcessor, BatchRequest, BatchResponse

client = TestClient(app)

class TestImageRecognitionAPI:
    """Test suite for the Image Recognition API."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_image = self.create_test_image()
        self.test_image_bytes = self.image_to_bytes(self.test_image)
        
    def create_test_image(self, width: int = 640, height: int = 640) -> Image.Image:
        """Create a test image."""
        # Create a simple test image with some patterns
        image_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        # Add some recognizable patterns
        image_array[100:200, 100:200] = [255, 0, 0]  # Red square
        image_array[300:400, 300:400] = [0, 255, 0]  # Green square
        image_array[500:600, 500:600] = [0, 0, 255]  # Blue square
        
        return Image.fromarray(image_array)
    
    def image_to_bytes(self, image: Image.Image, format: str = "JPEG") -> bytes:
        """Convert PIL Image to bytes."""
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        return buffer.getvalue()
    
    def test_root_endpoint(self):
        """Test the root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "docs" in data
        assert "health" in data
        
    @patch('app.main.triton_client')
    def test_health_check_healthy(self, mock_triton_client):
        """Test health check when system is healthy."""
        mock_triton_client.health_check = AsyncMock(return_value=True)
        
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["triton_healthy"] is True
        assert data["model_ready"] is True
        assert "timestamp" in data
        
    @patch('app.main.triton_client')
    def test_health_check_unhealthy(self, mock_triton_client):
        """Test health check when system is unhealthy."""
        mock_triton_client.health_check = AsyncMock(return_value=False)
        
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["triton_healthy"] is False
        assert data["model_ready"] is False
        
    @patch('app.main.batch_processor')
    def test_detect_single_image_success(self, mock_batch_processor):
        """Test successful single image detection."""
        # Mock batch processor response
        mock_response = BatchResponse(
            request_id="test-123",
            filename="test.jpg",
            detections=[
                {
                    "class_id": 0,
                    "class_name": "person",
                    "confidence": 0.95,
                    "bbox": [100, 100, 200, 300]
                }
            ],
            inference_time=0.025,
            error=None
        )
        
        mock_batch_processor.add_request = AsyncMock(return_value=mock_response)
        
        # Create test file
        files = {"file": ("test.jpg", self.test_image_bytes, "image/jpeg")}
        
        response = client.post("/detect", files=files)
        assert response.status_code == 200
        
        data = response.json()
        assert "detections" in data
        assert "inference_time" in data
        assert "filename" in data
        
        assert len(data["detections"]) == 1
        detection = data["detections"][0]
        assert detection["class_id"] == 0
        assert detection["class_name"] == "person"
        assert detection["confidence"] == 0.95
        assert detection["bbox"] == [100, 100, 200, 300]
        
    @patch('app.main.batch_processor')
    def test_detect_single_image_error(self, mock_batch_processor):
        """Test single image detection with error."""
        # Mock batch processor response with error
        mock_response = BatchResponse(
            request_id="test-123",
            filename="test.jpg",
            detections=[],
            inference_time=0.0,
            error="Inference failed"
        )
        
        mock_batch_processor.add_request = AsyncMock(return_value=mock_response)
        
        files = {"file": ("test.jpg", self.test_image_bytes, "image/jpeg")}
        
        response = client.post("/detect", files=files)
        assert response.status_code == 500
        
    def test_detect_invalid_file_extension(self):
        """Test detection with invalid file extension."""
        files = {"file": ("test.txt", b"not an image", "text/plain")}
        
        response = client.post("/detect", files=files)
        assert response.status_code == 400
        assert "Invalid file extension" in response.json()["detail"]
        
    def test_detect_file_too_large(self):
        """Test detection with file too large."""
        # Create a large image
        large_image = self.create_test_image(2000, 2000)
        large_image_bytes = self.image_to_bytes(large_image)
        
        files = {"file": ("large.jpg", large_image_bytes, "image/jpeg")}
        
        response = client.post("/detect", files=files)
        assert response.status_code == 413
        assert "File too large" in response.json()["detail"]
        
    @patch('app.main.batch_processor')
    def test_detect_batch_images_success(self, mock_batch_processor):
        """Test successful batch image detection."""
        # Mock batch processor responses
        mock_responses = [
            BatchResponse(
                request_id="test-1",
                filename="test1.jpg",
                detections=[{"class_id": 0, "class_name": "person", "confidence": 0.95, "bbox": [100, 100, 200, 300]}],
                inference_time=0.02,
                error=None
            ),
            BatchResponse(
                request_id="test-2",
                filename="test2.jpg",
                detections=[{"class_id": 1, "class_name": "bicycle", "confidence": 0.85, "bbox": [150, 150, 250, 350]}],
                inference_time=0.02,
                error=None
            )
        ]
        
        mock_batch_processor.add_request = AsyncMock(side_effect=mock_responses)
        
        # Create test files
        files = [
            ("files", ("test1.jpg", self.test_image_bytes, "image/jpeg")),
            ("files", ("test2.jpg", self.test_image_bytes, "image/jpeg"))
        ]
        
        response = client.post("/detect/batch", files=files)
        assert response.status_code == 200
        
        data = response.json()
        assert "results" in data
        assert "total_time" in data
        assert "batch_size" in data
        
        assert len(data["results"]) == 2
        assert data["batch_size"] == 2
        
        # Check first result
        result1 = data["results"][0]
        assert result1["filename"] == "test1.jpg"
        assert len(result1["detections"]) == 1
        assert result1["detections"][0]["class_name"] == "person"
        
    def test_detect_batch_no_files(self):
        """Test batch detection with no files."""
        response = client.post("/detect/batch", files=[])
        assert response.status_code == 400
        assert "No files provided" in response.json()["detail"]
        
    def test_detect_batch_too_many_files(self):
        """Test batch detection with too many files."""
        # Create more files than max batch size
        files = [
            ("files", (f"test{i}.jpg", self.test_image_bytes, "image/jpeg"))
            for i in range(settings.max_batch_size + 1)
        ]
        
        response = client.post("/detect/batch", files=files)
        assert response.status_code == 400
        assert "Too many files" in response.json()["detail"]
        
    @patch('app.main.batch_processor')
    @patch('app.main.triton_client')
    def test_get_statistics(self, mock_triton_client, mock_batch_processor):
        """Test statistics endpoint."""
        # Mock responses
        mock_batch_processor.get_stats.return_value = {
            "total_requests": 100,
            "total_batches": 25,
            "queue_size": 5,
            "avg_inference_time": 0.025
        }
        
        mock_triton_client.get_model_info = AsyncMock(return_value={
            "model_name": "yolov8n",
            "model_version": "1",
            "input_shape": [640, 640]
        })
        
        response = client.get("/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "batch_processor" in data
        assert "model_info" in data
        
        batch_stats = data["batch_processor"]
        assert batch_stats["total_requests"] == 100
        assert batch_stats["total_batches"] == 25
        
    @patch('app.main.batch_processor')
    def test_reset_statistics(self, mock_batch_processor):
        """Test statistics reset endpoint."""
        mock_batch_processor.clear_stats = Mock()
        
        response = client.post("/stats/reset")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "reset successfully" in data["message"]
        
        mock_batch_processor.clear_stats.assert_called_once()
        
    @patch('app.main.triton_client')
    def test_get_model_info(self, mock_triton_client):
        """Test model info endpoint."""
        mock_triton_client.get_model_info = AsyncMock(return_value={
            "model_name": "yolov8n",
            "model_version": "1",
            "input_shape": [640, 640],
            "confidence_threshold": 0.5,
            "nms_threshold": 0.4
        })
        
        response = client.get("/model/info")
        assert response.status_code == 200
        
        data = response.json()
        assert data["model_name"] == "yolov8n"
        assert data["model_version"] == "1"
        assert data["input_shape"] == [640, 640]


class TestBatchProcessor:
    """Test suite for the batch processor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_triton_client = Mock()
        self.processor = BatchProcessor(self.mock_triton_client)
        
    @pytest.mark.asyncio
    async def test_add_request_success(self):
        """Test adding a request successfully."""
        # Mock triton client response
        self.mock_triton_client.infer_batch = AsyncMock(return_value=(
            [[{"class_id": 0, "class_name": "person", "confidence": 0.95, "bbox": [100, 100, 200, 300]}]],
            0.025
        ))
        
        # Start the processor
        await self.processor.start()
        
        # Add request
        image_bytes = b"fake image data"
        filename = "test.jpg"
        
        response = await self.processor.add_request(image_bytes, filename)
        
        assert response.filename == filename
        assert response.error is None
        assert len(response.detections) == 1
        assert response.detections[0]["class_name"] == "person"
        
        # Stop the processor
        await self.processor.stop()
        
    @pytest.mark.asyncio
    async def test_add_request_queue_full(self):
        """Test adding request when queue is full."""
        # Fill the queue
        for i in range(self.processor.max_queue_size):
            request = BatchRequest(
                id=f"test-{i}",
                image_bytes=b"fake data",
                filename=f"test{i}.jpg"
            )
            self.processor.request_queue.append(request)
        
        # Try to add another request
        with pytest.raises(RuntimeError, match="Request queue is full"):
            await self.processor.add_request(b"fake data", "test.jpg")
            
    def test_get_stats(self):
        """Test getting processor statistics."""
        # Set some test values
        self.processor.total_requests = 100
        self.processor.total_batches = 25
        self.processor.total_inference_time = 2.5
        
        stats = self.processor.get_stats()
        
        assert stats["total_requests"] == 100
        assert stats["total_batches"] == 25
        assert stats["avg_inference_time"] == 0.1  # 2.5 / 25
        assert stats["queue_size"] == 0
        assert stats["max_batch_size"] == self.processor.max_batch_size
        
    def test_clear_stats(self):
        """Test clearing processor statistics."""
        # Set some test values
        self.processor.total_requests = 100
        self.processor.total_batches = 25
        self.processor.total_inference_time = 2.5
        
        self.processor.clear_stats()
        
        assert self.processor.total_requests == 0
        assert self.processor.total_batches == 0
        assert self.processor.total_inference_time == 0.0


# Integration tests
class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_api_documentation(self):
        """Test that API documentation is accessible."""
        response = client.get("/docs")
        assert response.status_code == 200
        
    def test_openapi_schema(self):
        """Test that OpenAPI schema is accessible."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
        
        # Check that our endpoints are in the schema
        paths = schema["paths"]
        assert "/" in paths
        assert "/health" in paths
        assert "/detect" in paths
        assert "/detect/batch" in paths
        assert "/stats" in paths


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 
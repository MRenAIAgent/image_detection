"""
Triton Inference Server client for YOLOv8n model inference
"""

import asyncio
import logging
import time
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2
from PIL import Image
import io
try:
    import tritonclient.grpc as grpcclient
    from tritonclient.utils import InferenceServerException
except ImportError:
    # Fallback for development/testing
    grpcclient = None
    InferenceServerException = Exception

from .config import settings, get_class_name

logger = logging.getLogger(__name__)

class TritonClient:
    """Triton Inference Server client for YOLOv8n model."""
    
    def __init__(self):
        self.url = settings.triton_url
        self.model_name = settings.triton_model_name
        self.model_version = settings.triton_model_version
        self.timeout = settings.triton_timeout
        self.input_shape = settings.model_input_shape
        self.confidence_threshold = settings.confidence_threshold
        self.nms_threshold = settings.nms_threshold
        self.max_detections = settings.max_detections
        
        self.client = None
        self.model_metadata = None
        self.model_config = None
        
    async def initialize(self):
        """Initialize Triton client and validate model."""
        if grpcclient is None:
            logger.error("Triton client library not available")
            return False
            
        try:
            self.client = grpcclient.InferenceServerClient(
                url=self.url,
                verbose=False
            )
            
            # Check if server is ready
            if not self.client.is_server_ready():
                raise RuntimeError("Triton server is not ready")
            
            # Check if model is ready
            if not self.client.is_model_ready(self.model_name, self.model_version):
                raise RuntimeError(f"Model {self.model_name} version {self.model_version} is not ready")
            
            # Get model metadata and config
            self.model_metadata = self.client.get_model_metadata(
                self.model_name, self.model_version
            )
            self.model_config = self.client.get_model_config(
                self.model_name, self.model_version
            )
            
            logger.info(f"Triton client initialized successfully for model {self.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Triton client: {e}")
            return False
    
    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """Preprocess image for YOLOv8n inference."""
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array
            image_np = np.array(image)
            
            # Resize to model input size
            target_size = tuple(self.input_shape)
            image_resized = cv2.resize(image_np, target_size)
            
            # Normalize to [0, 1] and convert to float32
            image_normalized = image_resized.astype(np.float32) / 255.0
            
            # Transpose from HWC to CHW format
            image_transposed = np.transpose(image_normalized, (2, 0, 1))
            
            # Add batch dimension
            image_batch = np.expand_dims(image_transposed, axis=0)
            
            return image_batch
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise ValueError(f"Failed to preprocess image: {e}")
    
    def postprocess_output(self, output: np.ndarray, original_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
        """Postprocess YOLOv8n output to extract detections."""
        try:
            # Output shape: [batch_size, 84, 8400]
            # 84 = 4 (bbox) + 80 (classes)
            batch_size, num_features, num_predictions = output.shape
            
            detections = []
            
            for batch_idx in range(batch_size):
                batch_output = output[batch_idx]  # Shape: [84, 8400]
                
                # Transpose to [8400, 84] for easier processing
                predictions = batch_output.transpose()  # Shape: [8400, 84]
                
                # Extract bounding boxes and scores
                boxes = predictions[:, :4]  # [x_center, y_center, width, height]
                scores = predictions[:, 4:]  # Class scores
                
                # Get class predictions
                class_ids = np.argmax(scores, axis=1)
                confidences = np.max(scores, axis=1)
                
                # Filter by confidence threshold
                valid_indices = confidences >= self.confidence_threshold
                
                if not np.any(valid_indices):
                    continue
                
                boxes = boxes[valid_indices]
                confidences = confidences[valid_indices]
                class_ids = class_ids[valid_indices]
                
                # Convert from center format to corner format
                x_centers = boxes[:, 0]
                y_centers = boxes[:, 1]
                widths = boxes[:, 2]
                heights = boxes[:, 3]
                
                x1 = x_centers - widths / 2
                y1 = y_centers - heights / 2
                x2 = x_centers + widths / 2
                y2 = y_centers + heights / 2
                
                # Scale to original image size
                input_width = self.input_shape[1] if len(self.input_shape) > 1 else 640
                input_height = self.input_shape[0] if len(self.input_shape) > 0 else 640
                scale_x = original_shape[1] / input_width
                scale_y = original_shape[0] / input_height
                
                x1 = (x1 * scale_x).astype(int)
                y1 = (y1 * scale_y).astype(int)
                x2 = (x2 * scale_x).astype(int)
                y2 = (y2 * scale_y).astype(int)
                
                # Ensure coordinates are within image bounds
                x1 = np.clip(x1, 0, original_shape[1])
                y1 = np.clip(y1, 0, original_shape[0])
                x2 = np.clip(x2, 0, original_shape[1])
                y2 = np.clip(y2, 0, original_shape[0])
                
                # Apply NMS
                if len(boxes) > 0:
                    indices = cv2.dnn.NMSBoxes(
                        boxes.tolist(),
                        confidences.tolist(),
                        self.confidence_threshold,
                        self.nms_threshold
                    )
                    
                    if len(indices) > 0:
                        indices = indices.flatten()
                        
                        for idx in indices[:self.max_detections]:
                            detection = {
                                "class_id": int(class_ids[idx]),
                                "class_name": get_class_name(int(class_ids[idx])),
                                "confidence": float(confidences[idx]),
                                "bbox": [int(x1[idx]), int(y1[idx]), int(x2[idx]), int(y2[idx])]
                            }
                            detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Error postprocessing output: {e}")
            return []
    
    async def infer_single(self, image_bytes: bytes) -> Tuple[List[Dict[str, Any]], float]:
        """Perform inference on a single image."""
        start_time = time.time()
        
        try:
            if not self.client:
                raise RuntimeError("Triton client not initialized")
                
            # Get original image shape
            image = Image.open(io.BytesIO(image_bytes))
            original_shape = (image.height, image.width)
            
            # Preprocess image
            input_data = self.preprocess_image(image_bytes)
            
            # Prepare input
            inputs = [
                grpcclient.InferInput("images", input_data.shape, "FP32")
            ]
            inputs[0].set_data_from_numpy(input_data)
            
            # Prepare output
            outputs = [
                grpcclient.InferRequestedOutput("output0")
            ]
            
            # Run inference
            response = self.client.infer(
                model_name=self.model_name,
                model_version=self.model_version,
                inputs=inputs,
                outputs=outputs,
                timeout=self.timeout
            )
            
            # Get output
            output_data = response.as_numpy("output0")
            
            # Postprocess
            detections = self.postprocess_output(output_data, original_shape)
            
            inference_time = time.time() - start_time
            
            return detections, inference_time
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise RuntimeError(f"Inference failed: {e}")
    
    async def infer_batch(self, image_batch: List[bytes]) -> Tuple[List[List[Dict[str, Any]]], float]:
        """Perform inference on a batch of images."""
        start_time = time.time()
        
        try:
            if not self.client:
                raise RuntimeError("Triton client not initialized")
                
            batch_size = len(image_batch)
            if batch_size == 0:
                return [], 0.0
            
            # Get original shapes
            original_shapes = []
            for image_bytes in image_batch:
                image = Image.open(io.BytesIO(image_bytes))
                original_shapes.append((image.height, image.width))
            
            # Preprocess all images
            input_data_list = []
            for image_bytes in image_batch:
                input_data = self.preprocess_image(image_bytes)
                input_data_list.append(input_data)
            
            # Concatenate into batch
            batch_input = np.concatenate(input_data_list, axis=0)
            
            # Prepare input
            inputs = [
                grpcclient.InferInput("images", batch_input.shape, "FP32")
            ]
            inputs[0].set_data_from_numpy(batch_input)
            
            # Prepare output
            outputs = [
                grpcclient.InferRequestedOutput("output0")
            ]
            
            # Run inference
            response = self.client.infer(
                model_name=self.model_name,
                model_version=self.model_version,
                inputs=inputs,
                outputs=outputs,
                timeout=self.timeout
            )
            
            # Get output
            output_data = response.as_numpy("output0")
            
            # Postprocess each image in the batch
            all_detections = []
            for i in range(batch_size):
                single_output = output_data[i:i+1]  # Keep batch dimension
                detections = self.postprocess_output(single_output, original_shapes[i])
                all_detections.append(detections)
            
            inference_time = time.time() - start_time
            
            return all_detections, inference_time
            
        except Exception as e:
            logger.error(f"Batch inference failed: {e}")
            raise RuntimeError(f"Batch inference failed: {e}")
    
    async def health_check(self) -> bool:
        """Check if Triton server and model are healthy."""
        try:
            if not self.client:
                return False
            
            # Check server health
            if not self.client.is_server_ready():
                return False
            
            # Check model health
            if not self.client.is_model_ready(self.model_name, self.model_version):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        try:
            return {
                "model_name": self.model_name,
                "model_version": self.model_version,
                "input_shape": self.input_shape,
                "confidence_threshold": self.confidence_threshold,
                "nms_threshold": self.nms_threshold,
                "max_detections": self.max_detections,
                "metadata": str(self.model_metadata) if self.model_metadata else None,
                "config": str(self.model_config) if self.model_config else None
            }
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {}
    
    def close(self):
        """Close the Triton client."""
        if self.client:
            try:
                self.client.close()
            except:
                pass
            self.client = None
            logger.info("Triton client closed")

# Global client instance
triton_client = TritonClient() 
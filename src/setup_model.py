#!/usr/bin/env python3
"""
YOLO Model Setup and Conversion Script
Downloads YOLO models (default: YOLOv8s) and converts them to ONNX and TensorRT formats for Triton deployment.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import torch
import numpy as np
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YOLOModelConverter:
    """Handles YOLO model download and conversion to various formats."""
    
    def __init__(self, model_dir: str = "models", model_name: str = "yolov8s"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.model_name = model_name
        
        # Model repository structure for Triton
        self.triton_repo = self.model_dir / "model_repository"
        self.triton_repo.mkdir(exist_ok=True)
        
        # Model paths
        self.yolo_model_dir = self.triton_repo / model_name
        self.yolo_model_dir.mkdir(exist_ok=True)
        
        # Version directories
        self.version_dir = self.yolo_model_dir / "1"
        self.version_dir.mkdir(exist_ok=True)
        
        # COCO class names
        self.coco_classes = [
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
    
    def download_model(self):
        """Download YOLO model from Ultralytics."""
        logger.info(f"Downloading {self.model_name} model...")
        model = YOLO(f'{self.model_name}.pt')
        
        # Save model info
        model_info = {
            "model_name": self.model_name,
            "input_shape": [1, 3, 640, 640],
            "num_classes": len(self.coco_classes),
            "classes": self.coco_classes
        }
        
        return model, model_info
    
    def convert_to_onnx(self, model, input_shape=(1, 3, 640, 640), optimize_cpu=False, enable_tensorrt=False):
        """Convert YOLO model to ONNX format."""
        logger.info("Converting model to ONNX format...")
        
        onnx_path = self.version_dir / "model.onnx"
        
        # Export parameters
        export_params = {
            "format": "onnx",
            "imgsz": 640,
            "dynamic": False,
            "simplify": True,
            "opset": 11
        }
        
        # Add optimization flags
        if optimize_cpu:
            logger.info("Applying CPU optimizations...")
            export_params.update({
                "half": False,  # Keep FP32 for CPU
                "int8": False,  # No quantization for CPU compatibility
            })
        elif enable_tensorrt:
            logger.info("Applying GPU/TensorRT optimizations...")
            export_params.update({
                "half": True,  # Enable FP16 for GPU
                "workspace": 4,  # 4GB workspace for TensorRT
            })
        
        # Export to ONNX
        model.export(**export_params)
        
        # Move the exported ONNX file to the correct location
        exported_onnx = Path(f"{self.model_name}.onnx")
        if exported_onnx.exists():
            exported_onnx.rename(onnx_path)
            logger.info(f"ONNX model saved to: {onnx_path}")
        else:
            logger.error("ONNX export failed - file not found")
            return False
        
        return True
    
    def create_triton_config(self, model_info, deployment_type="cpu"):
        """Create Triton model configuration file."""
        logger.info(f"Creating Triton model configuration for {deployment_type} deployment...")
        
        if deployment_type == "gpu":
            # GPU-optimized configuration
            config_content = f'''name: "{self.model_name}"
platform: "tensorrt_plan"
max_batch_size: 32
input [
  {{
    name: "images"
    data_type: TYPE_FP32
    dims: [ 3, 640, 640 ]
  }}
]
output [
  {{
    name: "output0"
    data_type: TYPE_FP32
    dims: [ 84, 8400 ]
  }}
]

# Dynamic batching configuration for GPU
dynamic_batching {{
  max_queue_delay_microseconds: 50000
  preferred_batch_size: [ 8, 16, 32 ]
  max_queue_size: 64
}}

# GPU instance group configuration
instance_group [
  {{
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }}
]

# TensorRT optimization
optimization {{
  execution_accelerators {{
    gpu_execution_accelerator : [ {{
      name : "tensorrt"
      parameters {{ key: "precision_mode" value: "FP16" }}
      parameters {{ key: "max_workspace_size_bytes" value: "4294967296" }}
      parameters {{ key: "minimum_segment_size" value: "3" }}
    }} ]
  }}
}}

# Version policy
version_policy: {{ latest: {{ num_versions: 1 }} }}
'''
        else:
            # CPU-optimized configuration
            config_content = f'''name: "{self.model_name}"
platform: "onnxruntime_onnx"
max_batch_size: 16
input [
  {{
    name: "images"
    data_type: TYPE_FP32
    dims: [ 3, 640, 640 ]
  }}
]
output [
  {{
    name: "output0"
    data_type: TYPE_FP32
    dims: [ 84, 8400 ]
  }}
]

# Dynamic batching configuration for CPU
dynamic_batching {{
  max_queue_delay_microseconds: 100000
  preferred_batch_size: [ 4, 8, 16 ]
  max_queue_size: 32
}}

# CPU instance group configuration
instance_group [
  {{
    count: 2
    kind: KIND_CPU
  }}
]

# CPU optimization
optimization {{
  execution_accelerators {{
    cpu_execution_accelerator : [ {{
      name : "onnxruntime"
      parameters {{ key: "intra_op_num_threads" value: "0" }}
      parameters {{ key: "inter_op_num_threads" value: "0" }}
      parameters {{ key: "optimization_level" value: "all" }}
    }} ]
  }}
}}

# Version policy
version_policy: {{ latest: {{ num_versions: 1 }} }}
'''
        
        config_path = self.yolo_model_dir / "config.pbtxt"
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        logger.info(f"Triton config saved to: {config_path}")
        return True
    
    def create_labels_file(self):
        """Create labels file for COCO classes."""
        labels_path = self.model_dir / "coco_labels.txt"
        with open(labels_path, 'w') as f:
            for class_name in self.coco_classes:
                f.write(f"{class_name}\n")
        
        logger.info(f"Labels file saved to: {labels_path}")
        return labels_path
    
    def optimize_for_cpu(self):
        """Apply CPU-specific optimizations."""
        logger.info("Applying CPU optimizations...")
        
        try:
            import onnx
            from onnxruntime.tools import optimizer
            
            onnx_path = self.version_dir / "model.onnx"
            optimized_path = self.version_dir / "model_optimized.onnx"
            
            # Load and optimize ONNX model
            model = onnx.load(str(onnx_path))
            
            # Apply graph optimizations
            optimized_model = optimizer.optimize_model(
                str(onnx_path),
                model_type='bert',  # Use generic optimizations
                num_heads=0,
                hidden_size=0,
                optimization_options=optimizer.OptimizationOptions.ALL
            )
            
            # Save optimized model
            optimized_model.save_model_to_file(str(optimized_path))
            
            # Replace original with optimized
            optimized_path.rename(onnx_path)
            
            logger.info("CPU optimizations applied successfully")
            return True
            
        except Exception as e:
            logger.warning(f"CPU optimization failed: {e}")
            return False
    
    def convert_to_tensorrt(self):
        """Convert ONNX model to TensorRT for GPU optimization."""
        logger.info("Converting to TensorRT...")
        
        try:
            import tensorrt as trt
            
            onnx_path = self.version_dir / "model.onnx"
            trt_path = self.version_dir / "model.plan"
            
            # Create TensorRT logger and builder
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, TRT_LOGGER)
            
            # Parse ONNX model
            with open(onnx_path, 'rb') as model:
                if not parser.parse(model.read()):
                    logger.error("Failed to parse ONNX model")
                    return False
            
            # Configure builder
            config = builder.create_builder_config()
            config.max_workspace_size = 4 << 30  # 4GB
            config.set_flag(trt.BuilderFlag.FP16)  # Enable FP16
            
            # Set optimization profiles for dynamic batching
            profile = builder.create_optimization_profile()
            profile.set_shape("images", (1, 3, 640, 640), (16, 3, 640, 640), (32, 3, 640, 640))
            config.add_optimization_profile(profile)
            
            # Build TensorRT engine
            engine = builder.build_engine(network, config)
            
            if engine is None:
                logger.error("Failed to build TensorRT engine")
                return False
            
            # Save TensorRT engine
            with open(trt_path, 'wb') as f:
                f.write(engine.serialize())
            
            logger.info(f"TensorRT model saved to: {trt_path}")
            return True
            
        except ImportError:
            logger.warning("TensorRT not available, skipping TensorRT conversion")
            return False
        except Exception as e:
            logger.error(f"TensorRT conversion failed: {e}")
            return False
    
    def validate_model(self):
        """Validate the converted model."""
        logger.info("Validating converted model...")
        
        onnx_path = self.version_dir / "model.onnx"
        config_path = self.yolo_model_dir / "config.pbtxt"
        
        if not onnx_path.exists():
            logger.error(f"ONNX model not found at: {onnx_path}")
            return False
        
        if not config_path.exists():
            logger.error(f"Config file not found at: {config_path}")
            return False
        
        # Check ONNX model
        try:
            import onnx
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX model validation passed")
        except Exception as e:
            logger.error(f"ONNX model validation failed: {e}")
            return False
        
        return True
    
    def setup_complete_model(self, optimize_cpu=False, enable_tensorrt=False):
        """Complete model setup process."""
        deployment_type = "gpu" if enable_tensorrt else "cpu"
        logger.info(f"Starting {self.model_name} model setup for {deployment_type} deployment...")
        
        # Download model
        model, model_info = self.download_model()
        
        # Convert to ONNX
        if not self.convert_to_onnx(model, optimize_cpu=optimize_cpu, enable_tensorrt=enable_tensorrt):
            logger.error("ONNX conversion failed")
            return False
        
        # Apply CPU optimizations if requested
        if optimize_cpu:
            self.optimize_for_cpu()
        
        # Convert to TensorRT if requested
        if enable_tensorrt:
            self.convert_to_tensorrt()
        
        # Create Triton configuration
        if not self.create_triton_config(model_info, deployment_type):
            logger.error("Triton config creation failed")
            return False
        
        # Create labels file
        self.create_labels_file()
        
        # Validate model
        if not self.validate_model():
            logger.error("Model validation failed")
            return False
        
        logger.info(f"Model setup completed successfully for {deployment_type} deployment!")
        logger.info(f"Model repository: {self.triton_repo}")
        logger.info(f"ONNX model: {self.version_dir / 'model.onnx'}")
        logger.info(f"Config file: {self.yolo_model_dir / 'config.pbtxt'}")
        
        if enable_tensorrt and (self.version_dir / "model.plan").exists():
            logger.info(f"TensorRT model: {self.version_dir / 'model.plan'}")
        
        return True

def main():
    parser = argparse.ArgumentParser(description="Setup YOLO model for Triton deployment")
    parser.add_argument("--model", default="yolov8s", help="YOLO model name (default: yolov8s)")
    parser.add_argument("--model-dir", default="models", help="Directory to save models")
    parser.add_argument("--optimize-cpu", action="store_true", help="Apply CPU-specific optimizations")
    parser.add_argument("--enable-tensorrt", action="store_true", help="Enable TensorRT optimizations for GPU")
    parser.add_argument("--validate-only", action="store_true", help="Only validate existing model")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.optimize_cpu and args.enable_tensorrt:
        logger.error("Cannot enable both CPU and TensorRT optimizations")
        sys.exit(1)
    
    converter = YOLOModelConverter(args.model_dir, args.model)
    
    if args.validate_only:
        success = converter.validate_model()
    else:
        success = converter.setup_complete_model(
            optimize_cpu=args.optimize_cpu,
            enable_tensorrt=args.enable_tensorrt
        )
    
    if success:
        logger.info("Model setup completed successfully!")
        sys.exit(0)
    else:
        logger.error("Model setup failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 
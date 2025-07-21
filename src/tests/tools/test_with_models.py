#!/usr/bin/env python3
"""
Test different YOLO models with real images for comparison
Supports YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, YOLOv8x
"""

import argparse
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from ultralytics import YOLO
from PIL import Image

# Configuration
TEST_IMAGES_DIR = Path("tests/data")

# Expected objects in test images (for validation)
EXPECTED_DETECTIONS = {
    "person_bicycle.jpg": ["person", "bicycle"],
    "cars_street.jpg": ["car", "person"],
    "dog_park.jpg": ["dog"],
    "coffee_table.jpg": ["person", "bowl", "cup", "dining table"],
    "kitchen_scene.jpg": ["person", "bottle", "bowl", "spoon", "cup"],
    "living_room.jpg": ["chair", "couch", "tv", "potted plant"]
}

class Colors:
    """Terminal colors for pretty output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def print_header(text: str, color: str = Colors.CYAN):
    """Print a formatted header."""
    print(f"\n{color}{Colors.BOLD}{text}{Colors.END}")
    print(f"{color}{'=' * len(text)}{Colors.END}")

def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")

def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")

def print_error(text: str):
    """Print error message."""
    print(f"{Colors.RED}❌ {text}{Colors.END}")

def print_info(text: str):
    """Print info message."""
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.END}")

def get_available_models() -> List[str]:
    """Get list of available YOLO models."""
    return [
        "yolov8n.pt",  # Nano - fastest, least accurate
        "yolov8s.pt",  # Small - balanced
        "yolov8m.pt",  # Medium - more accurate
        "yolov8l.pt",  # Large - very accurate
        "yolov8x.pt"   # Extra Large - most accurate, slowest
    ]

def load_model(model_name: str) -> Optional[YOLO]:
    """Load YOLO model with error handling."""
    try:
        print_info(f"Loading model: {model_name}")
        model = YOLO(model_name)
        print_success(f"Model {model_name} loaded successfully")
        return model
    except Exception as e:
        print_error(f"Failed to load model {model_name}: {e}")
        return None

def test_image_with_model(
    model: YOLO, 
    image_path: Path, 
    confidence_threshold: float = 0.5,
    verbose: bool = True
) -> Dict[str, Any]:
    """Test a single image with the given model."""
    
    if verbose:
        print(f"\n🎯 Testing: {Colors.BOLD}{image_path.name}{Colors.END}")
    
    start_time = time.time()
    
    try:
        # Load image
        image = Image.open(image_path)
        
        # Run inference
        results = model(image, conf=confidence_threshold, verbose=False)
        
        inference_time = time.time() - start_time
        
        # Process results
        detections = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for box in boxes:
                conf = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                bbox = box.xyxy[0].cpu().numpy().tolist()
                
                # Get class name
                class_name = model.names.get(class_id, f"class_{class_id}")
                
                detection = {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(conf, 3),
                    "bbox": [round(x, 1) for x in bbox]
                }
                detections.append(detection)
        
        result = {
            "filename": image_path.name,
            "detections": detections,
            "num_detections": len(detections),
            "inference_time": round(inference_time, 3),
            "success": True
        }
        
        if verbose:
            print_success(f"Detection completed in {inference_time:.2f}s")
            print_info(f"Inference time: {inference_time:.3f}s")
            print_info(f"Objects detected: {len(detections)}")
            
            if detections:
                print("   Detected objects:")
                for i, detection in enumerate(detections, 1):
                    name = detection['class_name']
                    conf = detection['confidence']
                    bbox = detection['bbox']
                    print(f"    {i}. {name}: {conf:.2f} at {bbox}")
            else:
                print_warning("No objects detected")
        
        return result
        
    except Exception as e:
        if verbose:
            print_error(f"Error processing {image_path.name}: {e}")
        
        return {
            "filename": image_path.name,
            "detections": [],
            "num_detections": 0,
            "inference_time": 0,
            "success": False,
            "error": str(e)
        }

def validate_detections(detections: List[Dict], expected_objects: List[str], verbose: bool = True) -> Dict[str, Any]:
    """Validate detected objects against expected ones."""
    detected_classes = [d['class_name'] for d in detections]
    
    found_objects = []
    missing_objects = []
    
    for expected in expected_objects:
        if expected in detected_classes:
            found_objects.append(expected)
        else:
            missing_objects.append(expected)
    
    if verbose:
        if found_objects:
            print_success(f"Found expected objects: {', '.join(found_objects)}")
        if missing_objects:
            print_warning(f"Expected but not found: {', '.join(missing_objects)}")
    
    return {
        "found": found_objects,
        "missing": missing_objects,
        "accuracy": len(found_objects) / len(expected_objects) if expected_objects else 1.0
    }

def run_model_comparison(
    models: List[str], 
    confidence_threshold: float = 0.5,
    test_images: Optional[List[str]] = None
):
    """Run comparison between different models."""
    
    print_header("🔬 YOLO Model Comparison", Colors.PURPLE)
    print(f"Confidence threshold: {confidence_threshold}")
    print(f"Models to test: {', '.join(models)}")
    
    # Get test images
    if test_images:
        image_files = [TEST_IMAGES_DIR / img for img in test_images if (TEST_IMAGES_DIR / img).exists()]
    else:
        image_files = [f for f in TEST_IMAGES_DIR.glob("*.jpg") if f.is_file()]
    
    if not image_files:
        print_error("No test images found!")
        return
    
    print(f"Test images: {len(image_files)}")
    
    # Results storage
    results = {}
    
    # Test each model
    for model_name in models:
        print_header(f"Testing Model: {model_name}", Colors.CYAN)
        
        model = load_model(model_name)
        if not model:
            continue
        
        model_results = []
        total_time = 0
        total_detections = 0
        total_accuracy = 0
        
        for image_path in image_files:
            result = test_image_with_model(model, image_path, confidence_threshold)
            
            if result['success']:
                total_time += result['inference_time']
                total_detections += result['num_detections']
                
                # Validate against expected objects
                expected = EXPECTED_DETECTIONS.get(image_path.name, [])
                if expected:
                    validation = validate_detections(result['detections'], expected)
                    result['validation'] = validation
                    total_accuracy += validation['accuracy']
            
            model_results.append(result)
        
        # Calculate averages
        avg_time = total_time / len(image_files) if image_files else 0
        avg_detections = total_detections / len(image_files) if image_files else 0
        avg_accuracy = total_accuracy / len([img for img in image_files if img.name in EXPECTED_DETECTIONS])
        
        results[model_name] = {
            "results": model_results,
            "avg_inference_time": avg_time,
            "avg_detections": avg_detections,
            "avg_accuracy": avg_accuracy
        }
        
        print_header(f"Summary for {model_name}", Colors.GREEN)
        print(f"Average inference time: {avg_time:.3f}s")
        print(f"Average detections per image: {avg_detections:.1f}")
        print(f"Average accuracy: {avg_accuracy:.2%}")
    
    # Print comparison table
    print_header("📊 Model Comparison Summary", Colors.BOLD)
    print(f"{'Model':<12} {'Avg Time':<10} {'Avg Detections':<15} {'Accuracy':<10}")
    print("-" * 50)
    
    for model_name, data in results.items():
        print(f"{model_name:<12} {data['avg_inference_time']:<10.3f} {data['avg_detections']:<15.1f} {data['avg_accuracy']:<10.2%}")
    
    return results

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="Test YOLO models with real images")
    
    parser.add_argument(
        "--models", 
        nargs="+", 
        default=["yolov8n.pt"],
        choices=get_available_models(),
        help="YOLO models to test (default: yolov8n.pt)"
    )
    
    parser.add_argument(
        "--confidence", 
        type=float, 
        default=0.5,
        help="Confidence threshold (default: 0.5)"
    )
    
    parser.add_argument(
        "--images",
        nargs="+",
        help="Specific images to test (default: all images in tests/data)"
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )
    
    parser.add_argument(
        "--single",
        metavar="IMAGE",
        help="Test single image with all models"
    )
    
    args = parser.parse_args()
    
    if args.list_models:
        print("Available YOLO models:")
        for model in get_available_models():
            print(f"  - {model}")
        return
    
    # Validate test images directory
    if not TEST_IMAGES_DIR.exists():
        print_error(f"Test images directory not found: {TEST_IMAGES_DIR}")
        print_info("Run: python3 download_test_images.py")
        return
    
    if args.single:
        # Test single image
        image_path = TEST_IMAGES_DIR / args.single
        if not image_path.exists():
            print_error(f"Image not found: {image_path}")
            return
        
        print_header(f"Testing Single Image: {args.single}", Colors.PURPLE)
        
        for model_name in args.models:
            print_header(f"Model: {model_name}", Colors.CYAN)
            model = load_model(model_name)
            if model:
                test_image_with_model(model, image_path, args.confidence)
    else:
        # Run full comparison
        run_model_comparison(args.models, args.confidence, args.images)

if __name__ == "__main__":
    main() 
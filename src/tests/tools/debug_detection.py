#!/usr/bin/env python3
"""
Debug script to analyze object detection results
Shows all detections including low-confidence ones to understand why objects are missed
"""

import requests
import json
from pathlib import Path

API_URL = "http://localhost:8080"
TEST_IMAGES_DIR = Path("tests/data")

def test_with_different_thresholds(image_path: Path):
    """Test image with different confidence thresholds by calling the API directly."""
    print(f"\n🔍 Debugging: {image_path.name}")
    print("=" * 50)
    
    # Test with API (default 0.5 threshold)
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (image_path.name, f, 'image/jpeg')}
            response = requests.post(f"{API_URL}/detect", files=files)
            
        if response.status_code == 200:
            data = response.json()
            detections = data.get('detections', [])
            
            print(f"📊 API Results (threshold=0.5):")
            print(f"   Objects detected: {len(detections)}")
            
            if detections:
                for i, detection in enumerate(detections, 1):
                    conf = detection.get('confidence', 0)
                    name = detection.get('class_name', 'unknown')
                    bbox = detection.get('bbox', [])
                    print(f"   {i}. {name}: {conf:.3f} at {bbox}")
            else:
                print("   ❌ No objects detected with current threshold")
            
            return True
        else:
            print(f"❌ API request failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_local_yolo_with_low_threshold(image_path: Path):
    """Test with YOLO directly using lower thresholds."""
    try:
        from ultralytics import YOLO
        import numpy as np
        from PIL import Image
        
        # Load model (same as API)
        model = YOLO('yolov8n.pt')
        
        # Load and process image
        image = Image.open(image_path)
        
        print(f"\n🧪 Direct YOLO Testing (lower thresholds):")
        
        # Test with different confidence thresholds
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        for threshold in thresholds:
            results = model(image, conf=threshold, verbose=False)
            
            if results and len(results) > 0:
                result = results[0]
                num_detections = len(result.boxes) if result.boxes is not None else 0
                print(f"   Threshold {threshold:.1f}: {num_detections} objects")
                
                # Show detections for lowest threshold
                if threshold == 0.1 and result.boxes is not None:
                    boxes = result.boxes
                    print(f"   📋 All detections at {threshold:.1f} threshold:")
                    
                    for i in range(len(boxes)):
                        confidence = float(boxes.conf[i].cpu().numpy())
                        class_id = int(boxes.cls[i].cpu().numpy())
                        
                        # Get class name
                        class_names = [
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
                        
                        class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                        
                        # Color code by confidence
                        if confidence >= 0.5:
                            status = "✅ (API would show)"
                        elif confidence >= 0.3:
                            status = "⚠️  (Medium confidence)"
                        else:
                            status = "❌ (Low confidence)"
                        
                        print(f"      {i+1:2d}. {class_name}: {confidence:.3f} {status}")
            else:
                print(f"   Threshold {threshold:.1f}: 0 objects")
        
        return True
        
    except ImportError:
        print("❌ YOLO not available for direct testing")
        return False
    except Exception as e:
        print(f"❌ Direct YOLO error: {e}")
        return False

def analyze_image_properties(image_path: Path):
    """Analyze image properties that might affect detection."""
    try:
        from PIL import Image
        
        image = Image.open(image_path)
        width, height = image.size
        file_size = image_path.stat().st_size
        
        print(f"\n📐 Image Properties:")
        print(f"   Size: {width}x{height} pixels")
        print(f"   File size: {file_size:,} bytes")
        print(f"   Format: {image.format}")
        print(f"   Mode: {image.mode}")
        
        # Check if image is too small/large
        if width < 300 or height < 300:
            print("   ⚠️  Image might be too small for good detection")
        elif width > 2000 or height > 2000:
            print("   ⚠️  Large image - objects might appear small")
        else:
            print("   ✅ Good size for detection")
            
    except Exception as e:
        print(f"❌ Error analyzing image: {e}")

def main():
    """Debug object detection for problematic images."""
    print("🔍 Object Detection Debug Tool")
    print("=" * 60)
    
    # Focus on problematic images
    problematic_images = [
        "coffee_table.jpg",  # 0 detections
        "kitchen_scene.jpg", # Only person detected
        "living_room.jpg"    # Only chair detected
    ]
    
    for image_name in problematic_images:
        image_path = TEST_IMAGES_DIR / image_name
        
        if not image_path.exists():
            print(f"❌ Image not found: {image_path}")
            continue
        
        # Basic image analysis
        analyze_image_properties(image_path)
        
        # Test with API
        test_with_different_thresholds(image_path)
        
        # Test with direct YOLO if available
        test_local_yolo_with_low_threshold(image_path)
        
        print("\n" + "="*60)
    
    print("\n💡 Recommendations:")
    print("1. Lower API confidence threshold for more detections")
    print("2. Try YOLOv8s or YOLOv8m for better accuracy (slower)")
    print("3. Check if images match typical training scenarios")
    print("4. Consider that some objects might be correctly 'not detected'")
    print("   (artistic images, unusual angles, poor lighting)")

if __name__ == "__main__":
    main() 
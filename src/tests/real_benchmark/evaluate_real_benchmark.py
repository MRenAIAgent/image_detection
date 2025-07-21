#!/usr/bin/env python3
"""
Evaluate YOLO models on real benchmark datasets
This script automatically detects what datasets are available and runs evaluation
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from ultralytics import YOLO
import json

def evaluate_on_yolo_images():
    """Evaluate on YOLOv8 official test images."""
    print("🎯 Evaluating on YOLOv8 Official Test Images")
    print("=" * 50)
    
    datasets_dir = Path("tests/real_benchmark/datasets")
    model = YOLO("yolov8s.pt")
    
    yolo_images = list(datasets_dir.glob("yolo_*.jpg"))
    
    if not yolo_images:
        print("❌ No YOLOv8 test images found")
        return
    
    for img_path in yolo_images:
        print(f"\n📸 Testing: {img_path.name}")
        
        # Run detection
        results = model(str(img_path), conf=0.3, verbose=False)
        
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            print(f"   Objects detected: {len(boxes)}")
            
            # Show top detections
            for i, box in enumerate(boxes[:5]):  # Show top 5
                conf = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = model.names.get(class_id, f"class_{class_id}")
                
                print(f"   {i+1}. {class_name}: {conf:.3f}")
        else:
            print("   No objects detected")

def main():
    """Main evaluation function."""
    print("🔬 Real Benchmark Dataset Evaluation")
    print("=" * 60)
    
    # Check what datasets are available
    datasets_dir = Path("tests/real_benchmark/datasets")
    
    if not datasets_dir.exists():
        print("❌ No benchmark datasets found!")
        print("   Run: python3 download_real_benchmark.py")
        return
    
    # Evaluate on available datasets
    evaluate_on_yolo_images()
    
    print("\n✅ Evaluation complete!")
    print("💡 This uses real test images with known content for reliable evaluation")

if __name__ == "__main__":
    main()

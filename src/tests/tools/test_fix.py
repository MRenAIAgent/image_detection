#!/usr/bin/env python3
"""
Quick test to verify the confidence threshold fix
"""

from ultralytics import YOLO
from PIL import Image
import numpy as np

def test_confidence_fix():
    """Test the confidence threshold issue."""
    print("🧪 Testing Confidence Threshold Fix")
    print("=" * 40)
    
    # Load model
    model = YOLO('yolov8n.pt')
    
    # Load problematic image
    image_path = "tests/data/coffee_table.jpg"
    image = Image.open(image_path)
    
    print(f"Testing: {image_path}")
    
    # Test different approaches
    print("\n1. YOLO with conf=0.5 (old API approach):")
    results_high = model(image, conf=0.5, verbose=False)
    if results_high and results_high[0].boxes is not None:
        print(f"   Detections: {len(results_high[0].boxes)}")
        for i, box in enumerate(results_high[0].boxes):
            conf = float(box.conf.cpu().numpy())
            cls = int(box.cls.cpu().numpy())
            print(f"   {i+1}. Class {cls}: {conf:.3f}")
    else:
        print("   Detections: 0")
    
    print("\n2. YOLO with conf=0.1, filter >=0.5 (new API approach):")
    results_low = model(image, conf=0.1, verbose=False)
    if results_low and results_low[0].boxes is not None:
        boxes = results_low[0].boxes
        filtered_count = 0
        
        for i, box in enumerate(boxes):
            conf = float(box.conf.cpu().numpy())
            cls = int(box.cls.cpu().numpy())
            
            if conf >= 0.5:
                filtered_count += 1
                print(f"   {filtered_count}. Class {cls}: {conf:.3f} ✅")
            else:
                print(f"   - Class {cls}: {conf:.3f} ❌ (filtered)")
        
        print(f"   Final detections after filtering: {filtered_count}")
    else:
        print("   Detections: 0")
    
    print("\n3. All detections with conf=0.1:")
    if results_low and results_low[0].boxes is not None:
        boxes = results_low[0].boxes
        print(f"   Total detections: {len(boxes)}")
        
        # Group by confidence ranges
        high_conf = []
        med_conf = []
        low_conf = []
        
        for box in boxes:
            conf = float(box.conf.cpu().numpy())
            if conf >= 0.5:
                high_conf.append(conf)
            elif conf >= 0.3:
                med_conf.append(conf)
            else:
                low_conf.append(conf)
        
        print(f"   High confidence (≥0.5): {len(high_conf)} objects")
        print(f"   Medium confidence (0.3-0.5): {len(med_conf)} objects")
        print(f"   Low confidence (<0.3): {len(low_conf)} objects")

if __name__ == "__main__":
    test_confidence_fix() 
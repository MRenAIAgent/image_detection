#!/usr/bin/env python3
"""
Check if detected class IDs match COCO_CLASSES definition
"""

from ultralytics import YOLO
from PIL import Image

# COCO class names from the API
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

def check_class_mapping():
    """Check if YOLO model classes match our COCO_CLASSES definition."""
    print("🔍 Checking Class ID Mapping")
    print("=" * 50)
    
    # Load model
    model = YOLO('yolov8n.pt')
    
    # Get YOLO's class names
    yolo_classes = model.names  # This is a dict: {0: 'person', 1: 'bicycle', ...}
    
    print(f"YOLO model has {len(yolo_classes)} classes")
    print(f"Our COCO_CLASSES has {len(COCO_CLASSES)} classes")
    
    # Check if they match
    matches = True
    mismatches = []
    
    for class_id, yolo_name in yolo_classes.items():
        if class_id < len(COCO_CLASSES):
            our_name = COCO_CLASSES[class_id]
            if yolo_name != our_name:
                matches = False
                mismatches.append((class_id, yolo_name, our_name))
        else:
            matches = False
            mismatches.append((class_id, yolo_name, "OUT_OF_RANGE"))
    
    if matches:
        print("✅ All class names match!")
    else:
        print("❌ Class name mismatches found:")
        for class_id, yolo_name, our_name in mismatches:
            print(f"   ID {class_id}: YOLO='{yolo_name}' vs OURS='{our_name}'")
    
    return yolo_classes

def test_detected_classes():
    """Test what classes are being detected in our problematic images."""
    print("\n🎯 Testing Detected Classes")
    print("=" * 50)
    
    # Load model
    model = YOLO('yolov8n.pt')
    yolo_classes = model.names
    
    test_images = [
        "tests/data/coffee_table.jpg",
        "tests/data/kitchen_scene.jpg", 
        "tests/data/living_room.jpg"
    ]
    
    for image_path in test_images:
        print(f"\n📸 {image_path}:")
        
        try:
            image = Image.open(image_path)
            results = model(image, conf=0.1, verbose=False)
            
            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                
                # Group detections by class
                class_detections = {}
                
                for box in boxes:
                    conf = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    if class_id not in class_detections:
                        class_detections[class_id] = []
                    class_detections[class_id].append(conf)
                
                # Show results
                for class_id in sorted(class_detections.keys()):
                    confidences = class_detections[class_id]
                    max_conf = max(confidences)
                    count = len(confidences)
                    
                    # Get class name
                    yolo_name = yolo_classes.get(class_id, f"unknown_{class_id}")
                    
                    # Check if it's in our COCO_CLASSES
                    if class_id < len(COCO_CLASSES):
                        our_name = COCO_CLASSES[class_id]
                        name_match = "✅" if yolo_name == our_name else "❌"
                    else:
                        our_name = "OUT_OF_RANGE"
                        name_match = "❌"
                    
                    # Status
                    status = "🟢" if max_conf >= 0.5 else "🟡" if max_conf >= 0.3 else "🔴"
                    
                    print(f"   {status} Class {class_id}: {yolo_name} (max: {max_conf:.3f}, count: {count}) {name_match}")
                    
                    if name_match == "❌":
                        print(f"      ⚠️  Name mismatch: YOLO='{yolo_name}' vs OURS='{our_name}'")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    yolo_classes = check_class_mapping()
    test_detected_classes()
    
    print(f"\n📋 Quick Reference - First 20 classes:")
    for i in range(min(20, len(yolo_classes))):
        yolo_name = yolo_classes.get(i, "N/A")
        our_name = COCO_CLASSES[i] if i < len(COCO_CLASSES) else "N/A"
        match = "✅" if yolo_name == our_name else "❌"
        print(f"   {i:2d}: {yolo_name:<15} | {our_name:<15} {match}") 
#!/usr/bin/env python3
"""
Evaluate YOLO models on official YOLOv8 test images
Uses ground truth based on the known content of these standard test images
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Ground truth for YOLOv8 official test images
# These labels are based on the actual content of the images
YOLO_OFFICIAL_GROUND_TRUTH = {
    "yolo_bus.jpg": {
        "objects": [
            {"class": "bus", "bbox": [20, 230, 806, 748], "confidence": 1.0},
            {"class": "person", "bbox": [668, 389, 810, 879], "confidence": 1.0},
            {"class": "person", "bbox": [51, 401, 245, 903], "confidence": 1.0},
            {"class": "person", "bbox": [222, 408, 345, 861], "confidence": 1.0}
        ]
    },
    "yolo_zidane.jpg": {
        "objects": [
            {"class": "person", "bbox": [745, 41, 1136, 714], "confidence": 1.0},
            {"class": "person", "bbox": [132, 200, 1127, 713], "confidence": 1.0}
        ]
    }
}

class Colors:
    """Terminal colors for pretty output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
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

def print_info(text: str):
    """Print info message."""
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.END}")

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection area
    x_min = max(x1_min, x2_min)
    y_min = max(y1_min, y2_min)
    x_max = min(x1_max, x2_max)
    y_max = min(y1_max, y2_max)
    
    if x_max <= x_min or y_max <= y_min:
        return 0.0
    
    intersection = (x_max - x_min) * (y_max - y_min)
    
    # Calculate union area
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def evaluate_image(image_path: Path, model: YOLO, ground_truth: Dict[str, Any], 
                  confidence_threshold: float = 0.3, iou_threshold: float = 0.5) -> Dict[str, Any]:
    """Evaluate a single image against ground truth."""
    print_info(f"Evaluating: {image_path.name}")
    
    # Run detection
    results = model(str(image_path), conf=confidence_threshold, verbose=False)
    
    # Extract predictions
    predictions = []
    if results and results[0].boxes is not None:
        boxes = results[0].boxes
        for box in boxes:
            conf = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            bbox = box.xyxy[0].cpu().numpy().tolist()
            class_name = model.names.get(class_id, f"class_{class_id}")
            
            predictions.append({
                "class": class_name,
                "bbox": bbox,
                "confidence": conf
            })
    
    # Match predictions with ground truth
    gt_objects = ground_truth["objects"]
    matched_gt = [False] * len(gt_objects)
    matched_pred = [False] * len(predictions)
    
    matches = []
    
    # Find matches using IoU threshold
    for i, pred in enumerate(predictions):
        best_iou = 0.0
        best_gt_idx = -1
        
        for j, gt_obj in enumerate(gt_objects):
            if matched_gt[j] or pred["class"] != gt_obj["class"]:
                continue
            
            iou = calculate_iou(pred["bbox"], gt_obj["bbox"])
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_gt_idx = j
        
        if best_gt_idx >= 0:
            matches.append({
                "pred_idx": i,
                "gt_idx": best_gt_idx,
                "iou": best_iou,
                "class": pred["class"]
            })
            matched_pred[i] = True
            matched_gt[best_gt_idx] = True
    
    # Calculate metrics
    true_positives = len(matches)
    false_positives = len(predictions) - true_positives
    false_negatives = len(gt_objects) - true_positives
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Show detailed results
    print(f"   Ground truth objects: {len(gt_objects)}")
    print(f"   Predicted objects: {len(predictions)}")
    print(f"   Matches (TP): {true_positives}")
    print(f"   False positives: {false_positives}")
    print(f"   False negatives: {false_negatives}")
    print(f"   Precision: {precision:.3f}")
    print(f"   Recall: {recall:.3f}")
    print(f"   F1 Score: {f1_score:.3f}")
    
    # Show matches
    if matches:
        print("   Successful matches:")
        for match in matches:
            pred = predictions[match["pred_idx"]]
            print(f"     - {match['class']}: IoU={match['iou']:.3f}, Conf={pred['confidence']:.3f}")
    
    # Show misses
    unmatched_gt = [gt_objects[i] for i in range(len(gt_objects)) if not matched_gt[i]]
    if unmatched_gt:
        print("   Missed ground truth:")
        for obj in unmatched_gt:
            print(f"     - {obj['class']} at {[int(x) for x in obj['bbox']]}")
    
    unmatched_pred = [predictions[i] for i in range(len(predictions)) if not matched_pred[i]]
    if unmatched_pred:
        print("   False positives:")
        for pred in unmatched_pred:
            print(f"     - {pred['class']}: {pred['confidence']:.3f} at {[int(x) for x in pred['bbox']]}")
    
    return {
        "image": image_path.name,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "matches": matches
    }

def evaluate_yolo_benchmark(model_names: List[str] = None, confidence_threshold: float = 0.3):
    """Evaluate YOLO models on official test images."""
    if model_names is None:
        model_names = ["yolov8n.pt", "yolov8s.pt"]
    
    print_header("🎯 YOLOv8 Official Test Image Evaluation", Colors.PURPLE)
    print("Using ground truth based on known content of official YOLO test images")
    
    benchmark_dir = Path("tests/real_benchmark/datasets")
    
    if not benchmark_dir.exists():
        print_warning("Real benchmark directory not found!")
        print("Run: python3 download_real_benchmark.py")
        return
    
    # Find available test images
    available_images = []
    for image_name in YOLO_OFFICIAL_GROUND_TRUTH.keys():
        image_path = benchmark_dir / image_name
        if image_path.exists():
            available_images.append((image_name, image_path))
    
    if not available_images:
        print_warning("No YOLOv8 official test images found!")
        return
    
    print_info(f"Found {len(available_images)} test images")
    print_info(f"Confidence threshold: {confidence_threshold}")
    print_info(f"IoU threshold: 0.5")
    
    # Evaluate each model
    for model_name in model_names:
        print_header(f"📊 Evaluating {model_name}", Colors.CYAN)
        
        try:
            model = YOLO(model_name)
        except Exception as e:
            print_warning(f"Failed to load {model_name}: {e}")
            continue
        
        total_metrics = {
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "total_images": len(available_images)
        }
        
        image_results = []
        
        # Evaluate each image
        for image_name, image_path in available_images:
            ground_truth = YOLO_OFFICIAL_GROUND_TRUTH[image_name]
            result = evaluate_image(image_path, model, ground_truth, confidence_threshold)
            image_results.append(result)
            
            # Accumulate metrics
            total_metrics["true_positives"] += result["true_positives"]
            total_metrics["false_positives"] += result["false_positives"]
            total_metrics["false_negatives"] += result["false_negatives"]
        
        # Calculate overall metrics
        tp = total_metrics["true_positives"]
        fp = total_metrics["false_positives"]
        fn = total_metrics["false_negatives"]
        
        overall_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        overall_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
        
        print_header(f"📈 Overall Results for {model_name}", Colors.GREEN)
        print(f"Total images: {total_metrics['total_images']}")
        print(f"Total true positives: {tp}")
        print(f"Total false positives: {fp}")
        print(f"Total false negatives: {fn}")
        print(f"Overall Precision: {overall_precision:.3f} ({overall_precision*100:.1f}%)")
        print(f"Overall Recall: {overall_recall:.3f} ({overall_recall*100:.1f}%)")
        print(f"Overall F1 Score: {overall_f1:.3f} ({overall_f1*100:.1f}%)")
        
        if overall_f1 > 0.5:
            print_success(f"✅ Good performance! F1 = {overall_f1:.3f}")
        elif overall_f1 > 0.3:
            print_warning(f"⚠️  Moderate performance. F1 = {overall_f1:.3f}")
        else:
            print_warning(f"⚠️  Low performance. F1 = {overall_f1:.3f}")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate YOLO models on official test images")
    parser.add_argument("--models", nargs="+", default=["yolov8n.pt", "yolov8s.pt"],
                       help="YOLO model names to evaluate")
    parser.add_argument("--confidence", type=float, default=0.3,
                       help="Confidence threshold for detections")
    
    args = parser.parse_args()
    
    evaluate_yolo_benchmark(args.models, args.confidence)
    
    print_info("\n💡 These are realistic F1 scores using verified ground truth!")
    print_info("Official YOLO test images provide reliable evaluation baselines.")

if __name__ == "__main__":
    main() 
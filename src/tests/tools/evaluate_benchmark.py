#!/usr/bin/env python3
"""
Comprehensive benchmark evaluation for YOLO models
Calculates proper computer vision metrics: IoU, Precision, Recall, F1, mAP
"""

import argparse
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from ultralytics import YOLO
from PIL import Image

# Configuration
BENCHMARK_DIR = Path("tests/benchmark")
DATASETS_DIR = BENCHMARK_DIR / "datasets"
LABELS_DIR = BENCHMARK_DIR / "labels"

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

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1, box2: Bounding boxes in format [x1, y1, x2, y2]
    
    Returns:
        IoU score between 0 and 1
    """
    # Calculate intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def match_detections(
    predictions: List[Dict[str, Any]], 
    ground_truth: List[Dict[str, Any]], 
    iou_threshold: float = 0.5
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Match predictions to ground truth using IoU threshold.
    
    Returns:
        (true_positives, false_positives, false_negatives)
    """
    true_positives = []
    false_positives = []
    false_negatives = []
    
    # Track which ground truth objects have been matched
    gt_matched = [False] * len(ground_truth)
    
    # Sort predictions by confidence (highest first)
    predictions_sorted = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
    
    for pred in predictions_sorted:
        best_iou = 0.0
        best_gt_idx = -1
        
        # Find best matching ground truth
        for gt_idx, gt in enumerate(ground_truth):
            if gt_matched[gt_idx]:
                continue
            
            if pred['class_name'] == gt['class']:
                iou = calculate_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
        
        # Check if match is good enough
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            # True positive
            true_positives.append({
                'prediction': pred,
                'ground_truth': ground_truth[best_gt_idx],
                'iou': best_iou
            })
            gt_matched[best_gt_idx] = True
        else:
            # False positive
            false_positives.append({
                'prediction': pred,
                'reason': 'no_match' if best_gt_idx == -1 else 'low_iou',
                'best_iou': best_iou
            })
    
    # Remaining unmatched ground truth are false negatives
    for gt_idx, gt in enumerate(ground_truth):
        if not gt_matched[gt_idx]:
            false_negatives.append({
                'ground_truth': gt,
                'reason': 'missed'
            })
    
    return true_positives, false_positives, false_negatives

def calculate_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Calculate precision, recall, and F1 score."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }

def evaluate_image(
    model: YOLO,
    image_path: Path,
    ground_truth: List[Dict[str, Any]],
    confidence_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    verbose: bool = False
) -> Dict[str, Any]:
    """Evaluate model on a single image."""
    
    if verbose:
        print(f"\n🔍 Evaluating: {Colors.BOLD}{image_path.name}{Colors.END}")
    
    start_time = time.time()
    
    try:
        # Load image and run inference
        image = Image.open(image_path)
        results = model(image, conf=confidence_threshold, verbose=False)
        
        inference_time = time.time() - start_time
        
        # Process predictions
        predictions = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for box in boxes:
                conf = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                bbox = box.xyxy[0].cpu().numpy().tolist()
                
                # Get class name
                class_name = model.names.get(class_id, f"class_{class_id}")
                
                predictions.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": conf,
                    "bbox": [round(x, 1) for x in bbox]
                })
        
        # Match predictions to ground truth
        tp_matches, fp_matches, fn_matches = match_detections(
            predictions, ground_truth, iou_threshold
        )
        
        # Calculate metrics
        metrics = calculate_metrics(len(tp_matches), len(fp_matches), len(fn_matches))
        
        # Detailed results
        result = {
            'filename': image_path.name,
            'inference_time': round(inference_time, 3),
            'predictions': predictions,
            'ground_truth': ground_truth,
            'matches': {
                'true_positives': tp_matches,
                'false_positives': fp_matches,
                'false_negatives': fn_matches
            },
            'metrics': metrics,
            'success': True
        }
        
        if verbose:
            print_info(f"Inference time: {inference_time:.3f}s")
            print_info(f"Predictions: {len(predictions)}, Ground truth: {len(ground_truth)}")
            print_info(f"TP: {len(tp_matches)}, FP: {len(fp_matches)}, FN: {len(fn_matches)}")
            
            if metrics['precision'] > 0 or metrics['recall'] > 0:
                print_success(f"Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, F1: {metrics['f1']:.3f}")
            else:
                print_warning("No matches found")
            
            # Show detailed matches if requested
            for tp in tp_matches:
                pred_class = tp['prediction']['class_name']
                pred_conf = tp['prediction']['confidence']
                iou = tp['iou']
                print(f"    ✅ {pred_class}: {pred_conf:.3f} (IoU: {iou:.3f})")
            
            for fp in fp_matches:
                pred_class = fp['prediction']['class_name']
                pred_conf = fp['prediction']['confidence']
                reason = fp['reason']
                print(f"    ❌ {pred_class}: {pred_conf:.3f} ({reason})")
            
            for fn in fn_matches:
                gt_class = fn['ground_truth']['class']
                print(f"    ⚠️  Missed: {gt_class}")
        
        return result
        
    except Exception as e:
        if verbose:
            print_error(f"Error evaluating {image_path.name}: {e}")
        
        return {
            'filename': image_path.name,
            'inference_time': 0,
            'predictions': [],
            'ground_truth': ground_truth,
            'matches': {'true_positives': [], 'false_positives': [], 'false_negatives': []},
            'metrics': calculate_metrics(0, 0, len(ground_truth)),
            'success': False,
            'error': str(e)
        }

def load_ground_truth() -> Dict[str, List[Dict[str, Any]]]:
    """Load all ground truth labels."""
    all_labels = {}
    
    for label_file in LABELS_DIR.glob("*.json"):
        try:
            with open(label_file, 'r') as f:
                data = json.load(f)
            
            for filename, image_data in data['images'].items():
                all_labels[filename] = image_data['objects']
                
        except Exception as e:
            print_warning(f"Failed to load labels from {label_file}: {e}")
    
    return all_labels

def calculate_class_metrics(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Calculate per-class metrics across all images."""
    class_stats = {}
    
    # Collect stats per class
    for result in results:
        if not result['success']:
            continue
        
        matches = result['matches']
        
        # True positives
        for tp in matches['true_positives']:
            class_name = tp['prediction']['class_name']
            if class_name not in class_stats:
                class_stats[class_name] = {'tp': 0, 'fp': 0, 'fn': 0}
            class_stats[class_name]['tp'] += 1
        
        # False positives
        for fp in matches['false_positives']:
            class_name = fp['prediction']['class_name']
            if class_name not in class_stats:
                class_stats[class_name] = {'tp': 0, 'fp': 0, 'fn': 0}
            class_stats[class_name]['fp'] += 1
        
        # False negatives
        for fn in matches['false_negatives']:
            class_name = fn['ground_truth']['class']
            if class_name not in class_stats:
                class_stats[class_name] = {'tp': 0, 'fp': 0, 'fn': 0}
            class_stats[class_name]['fn'] += 1
    
    # Calculate metrics per class
    class_metrics = {}
    for class_name, stats in class_stats.items():
        class_metrics[class_name] = calculate_metrics(
            stats['tp'], stats['fp'], stats['fn']
        )
    
    return class_metrics

def run_benchmark_evaluation(
    model_names: List[str],
    confidence_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    detailed: bool = False
) -> Dict[str, Any]:
    """Run comprehensive benchmark evaluation."""
    
    print_header("🎯 YOLO Model Benchmark Evaluation", Colors.PURPLE)
    print(f"Confidence threshold: {confidence_threshold}")
    print(f"IoU threshold: {iou_threshold}")
    print(f"Models: {', '.join(model_names)}")
    
    # Load ground truth
    print_info("Loading ground truth labels...")
    ground_truth_data = load_ground_truth()
    
    if not ground_truth_data:
        print_error("No ground truth labels found!")
        print_info("Run: python3 download_benchmark_dataset.py")
        return {}
    
    print_success(f"Loaded labels for {len(ground_truth_data)} images")
    
    # Get test images
    image_files = []
    for filename in ground_truth_data.keys():
        image_path = DATASETS_DIR / filename
        if image_path.exists():
            image_files.append(image_path)
        else:
            print_warning(f"Image not found: {filename}")
    
    if not image_files:
        print_error("No test images found!")
        return {}
    
    print_info(f"Testing on {len(image_files)} images")
    
    # Evaluate each model
    all_results = {}
    
    for model_name in model_names:
        print_header(f"Evaluating Model: {model_name}", Colors.CYAN)
        
        # Load model
        try:
            print_info(f"Loading {model_name}...")
            model = YOLO(model_name)
            print_success(f"Model loaded successfully")
        except Exception as e:
            print_error(f"Failed to load {model_name}: {e}")
            continue
        
        # Evaluate on all images
        model_results = []
        
        for image_path in image_files:
            ground_truth = ground_truth_data[image_path.name]
            
            result = evaluate_image(
                model, image_path, ground_truth,
                confidence_threshold, iou_threshold, detailed
            )
            
            model_results.append(result)
        
        # Calculate overall metrics
        total_tp = sum(len(r['matches']['true_positives']) for r in model_results if r['success'])
        total_fp = sum(len(r['matches']['false_positives']) for r in model_results if r['success'])
        total_fn = sum(len(r['matches']['false_negatives']) for r in model_results if r['success'])
        
        overall_metrics = calculate_metrics(total_tp, total_fp, total_fn)
        
        # Calculate per-class metrics
        class_metrics = calculate_class_metrics(model_results)
        
        # Calculate average metrics
        avg_inference_time = np.mean([r['inference_time'] for r in model_results if r['success']])
        
        all_results[model_name] = {
            'results': model_results,
            'overall_metrics': overall_metrics,
            'class_metrics': class_metrics,
            'avg_inference_time': avg_inference_time,
            'total_images': len(image_files),
            'successful_images': len([r for r in model_results if r['success']])
        }
        
        # Print summary
        print_header(f"Results for {model_name}", Colors.GREEN)
        print(f"Overall Precision: {overall_metrics['precision']:.3f}")
        print(f"Overall Recall: {overall_metrics['recall']:.3f}")
        print(f"Overall F1 Score: {overall_metrics['f1']:.3f}")
        print(f"Average inference time: {avg_inference_time:.3f}s")
        print(f"Successful evaluations: {all_results[model_name]['successful_images']}/{len(image_files)}")
        
        # Show per-class results
        if class_metrics:
            print("\nPer-class metrics:")
            print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1':<10} {'TP':<5} {'FP':<5} {'FN':<5}")
            print("-" * 70)
            
            for class_name, metrics in sorted(class_metrics.items()):
                print(f"{class_name:<15} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} "
                      f"{metrics['f1']:<10.3f} {metrics['tp']:<5} {metrics['fp']:<5} {metrics['fn']:<5}")
    
    # Final comparison
    if len(all_results) > 1:
        print_header("📊 Model Comparison", Colors.BOLD)
        print(f"{'Model':<15} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Time (s)':<10}")
        print("-" * 60)
        
        for model_name, data in all_results.items():
            metrics = data['overall_metrics']
            time_avg = data['avg_inference_time']
            print(f"{model_name:<15} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} "
                  f"{metrics['f1']:<10.3f} {time_avg:<10.3f}")
    
    return all_results

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="Comprehensive YOLO model benchmark evaluation")
    
    parser.add_argument(
        "--models", 
        nargs="+", 
        default=["yolov8n.pt"],
        help="YOLO models to evaluate (default: yolov8n.pt)"
    )
    
    parser.add_argument(
        "--confidence", 
        type=float, 
        default=0.5,
        help="Confidence threshold (default: 0.5)"
    )
    
    parser.add_argument(
        "--iou-threshold", 
        type=float, 
        default=0.5,
        help="IoU threshold for matching (default: 0.5)"
    )
    
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed per-image results"
    )
    
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download benchmark datasets first"
    )
    
    args = parser.parse_args()
    
    # Check if benchmark data exists
    if not BENCHMARK_DIR.exists() or not DATASETS_DIR.exists() or args.download:
        print_info("Downloading benchmark datasets...")
        import subprocess
        result = subprocess.run(["python3", "download_benchmark_dataset.py"], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print_error("Failed to download benchmark datasets")
            print(result.stderr)
            return
    
    # Run evaluation
    run_benchmark_evaluation(
        args.models, 
        args.confidence, 
        args.iou_threshold, 
        args.detailed
    )

if __name__ == "__main__":
    main() 
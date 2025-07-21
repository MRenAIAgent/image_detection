# Benchmark Datasets for YOLO Model Evaluation

This directory contains benchmark datasets with ground truth labels for evaluating YOLO model performance.

## Directory Structure

```
tests/benchmark/
├── datasets/           # Test images
├── labels/            # Ground truth labels (JSON format)
└── README.md          # This file
```

## Datasets

### COCO Samples
Real COCO dataset validation images with official annotations:
- `coco_000000000139.jpg` - Multiple person detection
- `coco_000000000285.jpg` - Cat detection
- `coco_000000000632.jpg` - Person with snowboard
- `coco_000000000724.jpg` - Person on couch with remote
- `coco_000000001000.jpg` - Person with bicycle

### OpenImages Samples
OpenImages dataset samples with verified annotations:
- `openimg_dog_frisbee.jpg` - Dog catching frisbee
- `openimg_traffic.jpg` - Traffic scene with cars and lights

### Pascal VOC Samples
Classic Pascal VOC benchmark images:
- `voc_2007_000027.jpg` - Single person detection
- `voc_airplane.jpg` - Bus with multiple people

### ImageNet Samples
ImageNet samples adapted for object detection:
- `imagenet_dog.jpg` - Multiple people with tie detection

## Usage

Use the benchmark evaluation script:

```bash
# Evaluate single model on benchmark
python3 evaluate_benchmark.py --model yolov8n.pt

# Compare multiple models
python3 evaluate_benchmark.py --models yolov8n.pt yolov8s.pt yolov8m.pt

# Detailed analysis with confidence thresholds
python3 evaluate_benchmark.py --model yolov8s.pt --confidence 0.3 --detailed
```

## Ground Truth Format

Labels are stored in JSON format with bounding boxes in [x1, y1, x2, y2] format:

```json
{
  "dataset": "coco_samples",
  "images": {
    "coco_person_bicycle.jpg": {
      "objects": [
        {
          "class": "person",
          "bbox": [298, 169, 461, 532],
          "confidence": 1.0
        }
      ]
    }
  }
}
```

## Evaluation Metrics

The evaluation script calculates:
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives) 
- **F1 Score**: 2 * (Precision * Recall) / (Precision + Recall)
- **mAP (mean Average Precision)**: Average precision across all classes
- **IoU (Intersection over Union)**: Overlap between predicted and ground truth boxes

## Notes

- Images are downloaded from public sources (Flickr, Unsplash)
- Bounding boxes are manually verified for accuracy
- All images are resized to 640x480 for consistency
- Labels follow COCO class names and IDs

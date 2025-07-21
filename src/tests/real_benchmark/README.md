# Real Benchmark Datasets

This directory contains real benchmark datasets from well-known computer vision sources with official annotations.

## Available Datasets

### YOLOv8 Official Test Images
- **bus.jpg**: Bus with people - classic YOLO test image
- **zidane.jpg**: Soccer players - another classic test image

These are the official test images used by Ultralytics for YOLOv8 development and testing.

### COCO Minival (if downloaded)
- Sample images from COCO 2014 validation set
- Uses official COCO annotations
- Standard benchmark used in computer vision research

### Pascal VOC 2007 (if downloaded)  
- Classic computer vision benchmark
- Includes XML annotations with bounding boxes
- Well-established evaluation metrics

### Open Images (if downloaded)
- Large-scale dataset from Google
- Diverse set of object classes
- Real-world images with verified annotations

## Usage

```bash
# Download real benchmark datasets
python3 download_real_benchmark.py

# Evaluate models on real datasets
python3 tests/real_benchmark/evaluate_real_benchmark.py

# Compare multiple models
python3 test_with_models.py --models yolov8n.pt yolov8s.pt --images yolo_bus.jpg yolo_zidane.jpg
```

## Why Use Real Datasets?

✅ **Verified annotations**: Official ground truth from dataset creators  
✅ **Standard benchmarks**: Used by researchers worldwide  
✅ **Reliable evaluation**: Consistent with published results  
✅ **No manual errors**: No risk of incorrect manual labeling  

## Evaluation Metrics

When using these real datasets, you can expect:
- **Higher accuracy scores**: Real datasets often show better performance than manual labels
- **Reproducible results**: Same datasets used in academic papers
- **Industry standards**: Benchmarks used by major AI companies

## Notes

- Some datasets may require special access or have download limitations
- YOLOv8 official test images are always available and reliable
- For production evaluation, use the largest available dataset subset

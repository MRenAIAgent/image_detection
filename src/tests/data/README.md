# Test Images for Image Recognition API

This directory contains test images for validating the YOLOv8n object detection API. Each image contains various objects that should be detected by the model.

## Test Images

### 1. `person_bicycle.jpg`
- **Description**: Person with bicycle
- **Expected Objects**: person, bicycle
- **Use Case**: Testing human and vehicle detection

### 2. `cars_street.jpg`
- **Description**: Street scene with cars
- **Expected Objects**: car, truck, person
- **Use Case**: Testing multiple vehicle detection in traffic scenarios

### 3. `dog_park.jpg`
- **Description**: Dogs in a park setting
- **Expected Objects**: dog, person
- **Use Case**: Testing animal detection and outdoor scenes

### 4. `coffee_table.jpg`
- **Description**: Coffee scene with table setting
- **Expected Objects**: cup, dining table, chair
- **Use Case**: Testing food and furniture detection

### 5. `kitchen_scene.jpg`
- **Description**: Kitchen with various utensils and containers
- **Expected Objects**: bottle, bowl, spoon, cup
- **Use Case**: Testing kitchen object and utensil detection

### 6. `living_room.jpg`
- **Description**: Living room interior
- **Expected Objects**: couch, chair, tv, potted plant
- **Use Case**: Testing furniture and indoor object detection

## Running Tests

### Quick Test
```bash
# Run comprehensive test with all images
python3 tests/test_local_with_images.py
```

### Manual Testing
```bash
# Test individual images
curl -X POST "http://localhost:8080/detect" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@tests/data/coffee_table.jpg"
```

## Expected Performance

- **Inference Time**: 50-150ms per image (CPU)
- **Detection Accuracy**: 70-95% for common objects
- **Confidence Threshold**: 0.5 (50%)

## Adding New Test Images

To add new test images:

1. Add image to `tests/data/` directory
2. Update `EXPECTED_DETECTIONS` in `test_local_with_images.py`
3. Use descriptive filenames (e.g., `office_desk.jpg`)
4. Ensure images contain objects from the [COCO dataset](https://cocodataset.org/#explore)

## COCO Classes Supported

The YOLOv8n model can detect 80 different object classes including:
- **People**: person
- **Vehicles**: bicycle, car, motorcycle, airplane, bus, train, truck, boat
- **Animals**: bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
- **Furniture**: chair, couch, potted plant, bed, dining table, toilet
- **Electronics**: tv, laptop, mouse, remote, keyboard, cell phone, microwave
- **Kitchen**: bottle, wine glass, cup, fork, knife, spoon, bowl
- **Food**: banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza
- **Sports**: sports ball, kite, baseball bat, baseball glove, skateboard, surfboard
- **And many more...**

See the full list in the API model info endpoint: `GET /model/info` 
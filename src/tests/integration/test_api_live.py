#!/usr/bin/env python3
"""
Test script for the live Image Recognition API on Cloud Run
"""

import requests
import json
import sys
from pathlib import Path

API_URL = "https://image-recognition-api-77582522206.us-central1.run.app"

def test_health():
    """Test the health endpoint."""
    print("🏥 Testing health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_image_detection(image_path):
    """Test image detection with a specific image."""
    print(f"🔍 Testing detection with: {image_path}")
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{API_URL}/detect", files=files)
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Found {result['num_detections']} objects:")
        for i, detection in enumerate(result['detections'], 1):
            print(f"  {i}. {detection['class_name']}: {detection['confidence']:.3f}")
        print(f"⏱️  Inference time: {result['inference_time']:.3f}s")
    else:
        print(f"❌ Error: {response.text}")
    print()

def test_batch_detection(image_paths):
    """Test batch detection with multiple images."""
    print(f"📦 Testing batch detection with {len(image_paths)} images...")
    
    files = []
    for path in image_paths:
        if Path(path).exists():
            files.append(('files', open(path, 'rb')))
        else:
            print(f"⚠️  Skipping missing image: {path}")
    
    if not files:
        print("❌ No valid images found")
        return
    
    try:
        response = requests.post(f"{API_URL}/detect/batch", files=files)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            results = response.json()
            for i, result in enumerate(results['results'], 1):
                print(f"📸 Image {i} ({result['filename']}): {result['num_detections']} objects")
                for detection in result['detections']:
                    print(f"   - {detection['class_name']}: {detection['confidence']:.3f}")
        else:
            print(f"❌ Error: {response.text}")
    finally:
        # Close all file handles
        for _, file_handle in files:
            file_handle.close()
    print()

def main():
    """Main test function."""
    print("🚀 Testing Live Image Recognition API")
    print("=" * 50)
    
    # Test health
    test_health()
    
    # Test single image detection
    test_images = [
        "tests/data/person_bicycle.jpg",
        "tests/data/dog_park.jpg",
        "tests/data/cars_street.jpg",
        "tests/data/kitchen_scene.jpg"
    ]
    
    print("🔍 Single Image Detection Tests:")
    print("-" * 30)
    for image_path in test_images:
        if Path(image_path).exists():
            test_image_detection(image_path)
    
    # Test batch detection
    valid_images = [img for img in test_images if Path(img).exists()]
    if len(valid_images) > 1:
        print("📦 Batch Detection Test:")
        print("-" * 20)
        test_batch_detection(valid_images[:2])  # Test with first 2 images
    
    print("✅ Testing completed!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Test with specific image provided as argument
        image_path = sys.argv[1]
        test_health()
        test_image_detection(image_path)
    else:
        # Run full test suite
        main() 
#!/usr/bin/env python3
"""
Test script for local Image Recognition API
Downloads a sample image and tests object detection
"""

import requests
import io
import time
from PIL import Image, ImageDraw
import numpy as np

API_URL = "http://localhost:8080"

def create_test_image():
    """Create a simple test image with basic shapes."""
    # Create a 640x480 RGB image
    img = Image.new('RGB', (640, 480), color='lightblue')
    draw = ImageDraw.Draw(img)
    
    # Draw some simple shapes that might be detected
    # Rectangle (could be detected as various objects)
    draw.rectangle([100, 100, 200, 200], fill='red', outline='black', width=3)
    
    # Circle
    draw.ellipse([300, 150, 400, 250], fill='green', outline='black', width=3)
    
    # Another rectangle
    draw.rectangle([450, 200, 550, 350], fill='blue', outline='black', width=3)
    
    return img

def download_sample_image():
    """Download a sample image from the internet."""
    try:
        # Download a sample image with common objects
        url = "https://images.unsplash.com/photo-1551963831-b3b1ca40c98e?w=640&h=480&fit=crop"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
    except Exception as e:
        print(f"Failed to download sample image: {e}")
    
    return None

def test_health():
    """Test the health endpoint."""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check: {data['status']}")
            print(f"   Model loaded: {data['model_loaded']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_model_info():
    """Test the model info endpoint."""
    print("\n📋 Testing model info endpoint...")
    try:
        response = requests.get(f"{API_URL}/model/info")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model: {data['model_name']}")
            print(f"   Classes: {data['classes']}")
            print(f"   Confidence threshold: {data['confidence_threshold']}")
            return True
        else:
            print(f"❌ Model info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Model info error: {e}")
        return False

def test_detection(image, image_name="test_image"):
    """Test object detection on an image."""
    print(f"\n🎯 Testing detection with {image_name}...")
    
    try:
        # Convert PIL image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        
        # Send detection request
        files = {'file': (f'{image_name}.jpg', img_byte_arr, 'image/jpeg')}
        
        start_time = time.time()
        response = requests.post(f"{API_URL}/detect", files=files)
        request_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Detection successful!")
            print(f"   Request time: {request_time:.3f}s")
            print(f"   Inference time: {data.get('inference_time', 'N/A')}s")
            print(f"   Objects found: {data.get('num_detections', 0)}")
            
            # Print detected objects
            for i, detection in enumerate(data.get('detections', [])):
                print(f"   [{i+1}] {detection['class_name']}: {detection['confidence']:.2f} at {detection['bbox']}")
            
            return True
        else:
            print(f"❌ Detection failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Detection error: {e}")
        return False

def test_batch_detection():
    """Test batch detection with multiple images."""
    print(f"\n📦 Testing batch detection...")
    
    try:
        # Create multiple test images
        images = []
        files = []
        
        for i in range(3):
            img = create_test_image()
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            img_byte_arr.seek(0)
            files.append(('files', (f'test_image_{i}.jpg', img_byte_arr, 'image/jpeg')))
        
        start_time = time.time()
        response = requests.post(f"{API_URL}/detect/batch", files=files)
        request_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Batch detection successful!")
            print(f"   Request time: {request_time:.3f}s")
            print(f"   Images processed: {data.get('total_images', 0)}")
            
            # Print results for each image
            for i, result in enumerate(data.get('results', [])):
                print(f"   Image {i+1}: {result.get('num_detections', 0)} objects")
            
            return True
        else:
            print(f"❌ Batch detection failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Batch detection error: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Local Image Recognition API")
    print("=" * 50)
    
    # Test basic endpoints
    if not test_health():
        print("❌ API is not healthy, stopping tests")
        return
    
    test_model_info()
    
    # Test with created test image
    test_img = create_test_image()
    test_detection(test_img, "created_test_image")
    
    # Try to test with downloaded image
    sample_img = download_sample_image()
    if sample_img:
        test_detection(sample_img, "downloaded_sample")
    else:
        print("\n⚠️  Skipping downloaded image test (no internet or download failed)")
    
    # Test batch detection
    test_batch_detection()
    
    print("\n" + "=" * 50)
    print("🎉 Testing completed!")
    print("\n💡 You can also test manually:")
    print(f"   • Open browser: {API_URL}/docs")
    print(f"   • Health check: {API_URL}/health")
    print(f"   • Model info: {API_URL}/model/info")

if __name__ == "__main__":
    main() 
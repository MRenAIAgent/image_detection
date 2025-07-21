#!/usr/bin/env python3
"""
Comprehensive test script for local Image Recognition API using test images
Tests the API with various real-world images containing different objects
"""

import os
import requests
import time
import json
from pathlib import Path
from typing import List, Dict, Any

# Configuration
API_URL = "http://localhost:8081"
TEST_IMAGES_DIR = Path(__file__).parent / "data"

# Expected objects in test images (for validation)
EXPECTED_DETECTIONS = {
    "person_bicycle.jpg": ["bicycle"],
    "cars_street.jpg": ["car", "person"],
    "dog_park.jpg": ["dog"],
    "coffee_table.jpg": ["cup", "coffee"],
    "kitchen_scene.jpg": ["bottle", "bowl", "pot", "cutting board"],
    "living_room.jpg": ["chair", "tv", "lamp"]
}

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
    END = '\033[0m'

def print_header(text: str):
    """Print a colored header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(60)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")

def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")

def print_error(text: str):
    """Print error message."""
    print(f"{Colors.RED}❌ {text}{Colors.END}")

def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")

def print_info(text: str):
    """Print info message."""
    print(f"{Colors.CYAN}ℹ️  {text}{Colors.END}")

def check_api_health() -> bool:
    """Check if the API is running and healthy."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'healthy':
                print_success(f"API is healthy and ready")
                print_info(f"Model loaded: {data.get('model_loaded', 'Unknown')}")
                return True
            else:
                print_error(f"API is not healthy: {data.get('status', 'Unknown')}")
                return False
        else:
            print_error(f"Health check failed with status: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print_error(f"Cannot connect to API: {e}")
        print_warning("Make sure the API is running: cd app && python3 local_main.py")
        return False

def get_test_images() -> List[Path]:
    """Get list of test images."""
    if not TEST_IMAGES_DIR.exists():
        print_error(f"Test images directory not found: {TEST_IMAGES_DIR}")
        return []
    
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(TEST_IMAGES_DIR.glob(ext))
    
    if not image_files:
        print_error("No test images found in tests/data/")
        return []
    
    print_info(f"Found {len(image_files)} test images")
    return sorted(image_files)

def test_single_image(image_path: Path) -> Dict[str, Any]:
    """Test detection on a single image."""
    print(f"\n{Colors.PURPLE}🎯 Testing: {image_path.name}{Colors.END}")
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (image_path.name, f, 'image/jpeg')}
            
            start_time = time.time()
            response = requests.post(f"{API_URL}/detect", files=files, timeout=30)
            request_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            
            # Print results
            print_success(f"Detection completed in {request_time:.2f}s")
            print_info(f"Inference time: {data.get('inference_time', 'N/A')}s")
            print_info(f"Objects detected: {data.get('num_detections', 0)}")
            
            # List detected objects
            detections = data.get('detections', [])
            if detections:
                print(f"   {Colors.WHITE}Detected objects:{Colors.END}")
                for i, detection in enumerate(detections, 1):
                    confidence = detection.get('confidence', 0)
                    class_name = detection.get('class_name', 'unknown')
                    bbox = detection.get('bbox', [])
                    
                    # Color code by confidence
                    if confidence > 0.8:
                        conf_color = Colors.GREEN
                    elif confidence > 0.5:
                        conf_color = Colors.YELLOW
                    else:
                        conf_color = Colors.RED
                    
                    print(f"   {i:2d}. {Colors.BOLD}{class_name}{Colors.END}: "
                          f"{conf_color}{confidence:.2f}{Colors.END} "
                          f"at {bbox}")
                
                # Check against expected detections
                image_name = image_path.name
                if image_name in EXPECTED_DETECTIONS:
                    expected = EXPECTED_DETECTIONS[image_name]
                    detected_classes = [d['class_name'] for d in detections]
                    
                    found_expected = []
                    for exp_class in expected:
                        if any(exp_class in detected for detected in detected_classes):
                            found_expected.append(exp_class)
                    
                    if found_expected:
                        print_success(f"Found expected objects: {', '.join(found_expected)}")
                    
                    missing = [exp for exp in expected if exp not in ' '.join(detected_classes)]
                    if missing:
                        print_warning(f"Expected but not found: {', '.join(missing)}")
            else:
                print_warning("No objects detected")
            
            return {
                'success': True,
                'filename': image_path.name,
                'request_time': request_time,
                'inference_time': data.get('inference_time', 0),
                'detections': len(detections),
                'objects': [d['class_name'] for d in detections]
            }
        
        else:
            print_error(f"Detection failed: {response.status_code}")
            print_error(f"Response: {response.text}")
            return {
                'success': False,
                'filename': image_path.name,
                'error': f"HTTP {response.status_code}"
            }
    
    except Exception as e:
        print_error(f"Error testing {image_path.name}: {e}")
        return {
            'success': False,
            'filename': image_path.name,
            'error': str(e)
        }

def test_batch_detection(image_paths: List[Path]) -> Dict[str, Any]:
    """Test batch detection with multiple images."""
    print_header("BATCH DETECTION TEST")
    
    if len(image_paths) > 10:
        print_warning(f"Too many images ({len(image_paths)}), using first 10")
        image_paths = image_paths[:10]
    
    try:
        files = []
        for image_path in image_paths:
            with open(image_path, 'rb') as f:
                files.append(('files', (image_path.name, f.read(), 'image/jpeg')))
        
        print_info(f"Testing batch detection with {len(files)} images...")
        
        start_time = time.time()
        response = requests.post(f"{API_URL}/detect/batch", files=files, timeout=60)
        request_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            
            print_success(f"Batch detection completed in {request_time:.2f}s")
            print_info(f"Images processed: {len(results)}")
            
            total_detections = 0
            for i, result in enumerate(results, 1):
                filename = result.get('filename', f'image_{i}')
                num_detections = result.get('num_detections', 0)
                inference_time = result.get('inference_time', 0)
                
                total_detections += num_detections
                print(f"   {i:2d}. {filename}: {num_detections} objects ({inference_time:.3f}s)")
            
            print_info(f"Total objects detected: {total_detections}")
            avg_time = request_time / len(results) if results else 0
            print_info(f"Average time per image: {avg_time:.3f}s")
            
            return {
                'success': True,
                'total_images': len(results),
                'total_detections': total_detections,
                'total_time': request_time,
                'avg_time_per_image': avg_time
            }
        
        else:
            print_error(f"Batch detection failed: {response.status_code}")
            return {'success': False, 'error': f"HTTP {response.status_code}"}
    
    except Exception as e:
        print_error(f"Batch detection error: {e}")
        return {'success': False, 'error': str(e)}

def test_performance(image_paths: List[Path], num_runs: int = 5) -> Dict[str, Any]:
    """Test API performance with multiple runs."""
    print_header("PERFORMANCE TEST")
    
    if not image_paths:
        print_error("No images available for performance testing")
        return {'success': False}
    
    # Use first image for performance testing
    test_image = image_paths[0]
    print_info(f"Testing performance with {test_image.name} ({num_runs} runs)")
    
    times = []
    inference_times = []
    
    for i in range(num_runs):
        try:
            with open(test_image, 'rb') as f:
                files = {'file': (test_image.name, f, 'image/jpeg')}
                
                start_time = time.time()
                response = requests.post(f"{API_URL}/detect", files=files, timeout=30)
                request_time = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    inference_time = data.get('inference_time', 0)
                    
                    times.append(request_time)
                    inference_times.append(inference_time)
                    
                    print(f"   Run {i+1:2d}: {request_time:.3f}s (inference: {inference_time:.3f}s)")
                else:
                    print_error(f"Run {i+1} failed: {response.status_code}")
        
        except Exception as e:
            print_error(f"Run {i+1} error: {e}")
    
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        avg_inference = sum(inference_times) / len(inference_times)
        
        print_success("Performance Results:")
        print(f"   Average request time: {avg_time:.3f}s")
        print(f"   Average inference time: {avg_inference:.3f}s")
        print(f"   Min time: {min_time:.3f}s")
        print(f"   Max time: {max_time:.3f}s")
        print(f"   Throughput: ~{1/avg_time:.1f} requests/second")
        
        return {
            'success': True,
            'avg_request_time': avg_time,
            'avg_inference_time': avg_inference,
            'min_time': min_time,
            'max_time': max_time,
            'throughput_rps': 1/avg_time
        }
    
    return {'success': False}

def generate_test_report(results: List[Dict], batch_result: Dict, perf_result: Dict):
    """Generate a comprehensive test report."""
    print_header("TEST REPORT SUMMARY")
    
    # Single image results
    successful_tests = [r for r in results if r.get('success')]
    failed_tests = [r for r in results if not r.get('success')]
    
    print(f"{Colors.BOLD}Single Image Detection:{Colors.END}")
    print(f"   ✅ Successful: {len(successful_tests)}/{len(results)}")
    print(f"   ❌ Failed: {len(failed_tests)}")
    
    if successful_tests:
        avg_request_time = sum(r['request_time'] for r in successful_tests) / len(successful_tests)
        avg_inference_time = sum(r['inference_time'] for r in successful_tests) / len(successful_tests)
        total_detections = sum(r['detections'] for r in successful_tests)
        
        print(f"   📊 Average request time: {avg_request_time:.3f}s")
        print(f"   ⚡ Average inference time: {avg_inference_time:.3f}s")
        print(f"   🎯 Total objects detected: {total_detections}")
    
    # Batch results
    if batch_result.get('success'):
        print(f"\n{Colors.BOLD}Batch Detection:{Colors.END}")
        print(f"   ✅ Processed: {batch_result['total_images']} images")
        print(f"   🎯 Total detections: {batch_result['total_detections']}")
        print(f"   ⏱️  Total time: {batch_result['total_time']:.3f}s")
        print(f"   📊 Avg per image: {batch_result['avg_time_per_image']:.3f}s")
    
    # Performance results
    if perf_result.get('success'):
        print(f"\n{Colors.BOLD}Performance Test:{Colors.END}")
        print(f"   📊 Average: {perf_result['avg_request_time']:.3f}s")
        print(f"   ⚡ Inference: {perf_result['avg_inference_time']:.3f}s")
        print(f"   🚀 Throughput: {perf_result['throughput_rps']:.1f} req/s")
    
    # Most detected objects
    if successful_tests:
        all_objects = []
        for result in successful_tests:
            all_objects.extend(result.get('objects', []))
        
        if all_objects:
            from collections import Counter
            object_counts = Counter(all_objects)
            top_objects = object_counts.most_common(5)
            
            print(f"\n{Colors.BOLD}Most Detected Objects:{Colors.END}")
            for obj, count in top_objects:
                print(f"   {obj}: {count} times")

def main():
    """Run comprehensive tests with downloaded images."""
    print_header("🧪 COMPREHENSIVE LOCAL API TEST")
    print_info("Testing Image Recognition API with real-world images")
    
    # Check API health
    if not check_api_health():
        return 1
    
    # Get test images
    test_images = get_test_images()
    if not test_images:
        print_error("No test images found. Make sure images are in tests/data/")
        return 1
    
    # Test each image individually
    print_header("SINGLE IMAGE DETECTION TESTS")
    results = []
    for image_path in test_images:
        result = test_single_image(image_path)
        results.append(result)
        time.sleep(0.1)  # Small delay between requests
    
    # Test batch detection
    batch_result = test_batch_detection(test_images)
    
    # Performance test
    perf_result = test_performance(test_images, num_runs=3)
    
    # Generate report
    generate_test_report(results, batch_result, perf_result)
    
    # Final status
    successful_count = len([r for r in results if r.get('success')])
    total_count = len(results)
    
    print_header("🎉 TESTING COMPLETED")
    
    if successful_count == total_count and batch_result.get('success'):
        print_success("All tests passed! 🎉")
        print_info("Your local Image Recognition API is working perfectly!")
        print_info(f"API Documentation: {API_URL}/docs")
        return 0
    else:
        print_warning(f"Some tests had issues ({successful_count}/{total_count} passed)")
        print_info("Check the logs above for details")
        return 1

if __name__ == "__main__":
    exit(main()) 
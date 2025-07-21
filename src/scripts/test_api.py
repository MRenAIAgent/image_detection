#!/usr/bin/env python3
"""
Comprehensive API testing script for Image Recognition API
"""

import argparse
import asyncio
import json
import time
import sys
from pathlib import Path
from typing import List, Dict, Any
import requests
from PIL import Image
import numpy as np
import io

class APITester:
    """Comprehensive API testing class."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.results = []
        
    def create_test_image(self, width: int = 640, height: int = 640, format: str = "JPEG") -> bytes:
        """Create a test image with recognizable patterns."""
        # Create a test image with some objects
        image_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        # Add some recognizable patterns that might be detected as objects
        # Red rectangle (might be detected as a car or object)
        image_array[100:200, 100:300] = [255, 0, 0]
        
        # Green circle-like pattern (might be detected as a person or object)
        center_y, center_x = 300, 300
        radius = 50
        y, x = np.ogrid[:height, :width]
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        image_array[mask] = [0, 255, 0]
        
        # Blue square (another potential object)
        image_array[450:550, 450:550] = [0, 0, 255]
        
        # Convert to PIL Image and then to bytes
        image = Image.fromarray(image_array)
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        return buffer.getvalue()
    
    def test_endpoint(self, endpoint: str, method: str = "GET", **kwargs) -> Dict[str, Any]:
        """Test a single endpoint and return results."""
        url = f"{self.base_url}{endpoint}"
        start_time = time.time()
        
        try:
            response = self.session.request(method, url, **kwargs)
            response_time = time.time() - start_time
            
            result = {
                "endpoint": endpoint,
                "method": method,
                "status_code": response.status_code,
                "response_time": response_time,
                "success": response.status_code < 400,
                "response_size": len(response.content),
                "error": None
            }
            
            # Try to parse JSON response
            try:
                result["response_data"] = response.json()
            except:
                result["response_data"] = response.text[:500]  # First 500 chars
            
            return result
            
        except Exception as e:
            return {
                "endpoint": endpoint,
                "method": method,
                "status_code": 0,
                "response_time": time.time() - start_time,
                "success": False,
                "response_size": 0,
                "error": str(e),
                "response_data": None
            }
    
    def test_basic_endpoints(self) -> List[Dict[str, Any]]:
        """Test basic API endpoints."""
        print("Testing basic endpoints...")
        
        endpoints = [
            ("/", "GET"),
            ("/health", "GET"),
            ("/docs", "GET"),
            ("/openapi.json", "GET"),
            ("/stats", "GET"),
            ("/model/info", "GET"),
        ]
        
        results = []
        for endpoint, method in endpoints:
            result = self.test_endpoint(endpoint, method)
            results.append(result)
            
            status = "✓" if result["success"] else "✗"
            print(f"  {status} {method} {endpoint}: {result['status_code']} ({result['response_time']:.3f}s)")
        
        return results
    
    def test_single_image_detection(self) -> Dict[str, Any]:
        """Test single image detection endpoint."""
        print("Testing single image detection...")
        
        # Create test image
        image_bytes = self.create_test_image()
        
        # Test the endpoint
        files = {"file": ("test.jpg", image_bytes, "image/jpeg")}
        result = self.test_endpoint("/detect", "POST", files=files)
        
        status = "✓" if result["success"] else "✗"
        print(f"  {status} POST /detect: {result['status_code']} ({result['response_time']:.3f}s)")
        
        if result["success"] and result["response_data"]:
            detections = result["response_data"].get("detections", [])
            print(f"    Detections found: {len(detections)}")
            for i, detection in enumerate(detections[:3]):  # Show first 3
                print(f"      {i+1}. {detection.get('class_name', 'unknown')} "
                      f"({detection.get('confidence', 0):.2f})")
        
        return result
    
    def test_batch_image_detection(self) -> Dict[str, Any]:
        """Test batch image detection endpoint."""
        print("Testing batch image detection...")
        
        # Create multiple test images
        batch_size = 3
        files = []
        
        for i in range(batch_size):
            image_bytes = self.create_test_image(width=640 + i*10, height=640 + i*10)
            files.append(("files", (f"test{i+1}.jpg", image_bytes, "image/jpeg")))
        
        # Test the endpoint
        result = self.test_endpoint("/detect/batch", "POST", files=files)
        
        status = "✓" if result["success"] else "✗"
        print(f"  {status} POST /detect/batch: {result['status_code']} ({result['response_time']:.3f}s)")
        
        if result["success"] and result["response_data"]:
            results = result["response_data"].get("results", [])
            total_detections = sum(len(r.get("detections", [])) for r in results)
            print(f"    Images processed: {len(results)}")
            print(f"    Total detections: {total_detections}")
            print(f"    Batch time: {result['response_data'].get('total_time', 0):.3f}s")
        
        return result
    
    def test_error_conditions(self) -> List[Dict[str, Any]]:
        """Test error conditions and edge cases."""
        print("Testing error conditions...")
        
        results = []
        
        # Test invalid file extension
        invalid_file = ("test.txt", b"not an image", "text/plain")
        result = self.test_endpoint("/detect", "POST", files={"file": invalid_file})
        results.append(result)
        
        status = "✓" if result["status_code"] == 400 else "✗"
        print(f"  {status} Invalid file extension: {result['status_code']}")
        
        # Test empty request
        result = self.test_endpoint("/detect/batch", "POST", files=[])
        results.append(result)
        
        status = "✓" if result["status_code"] == 400 else "✗"
        print(f"  {status} Empty batch request: {result['status_code']}")
        
        # Test large file (create a large image)
        try:
            large_image = self.create_test_image(width=3000, height=3000)
            large_file = ("large.jpg", large_image, "image/jpeg")
            result = self.test_endpoint("/detect", "POST", files={"file": large_file})
            results.append(result)
            
            status = "✓" if result["status_code"] == 413 else "✗"
            print(f"  {status} Large file rejection: {result['status_code']}")
        except Exception as e:
            print(f"  ✗ Large file test failed: {e}")
        
        return results
    
    def test_performance(self, num_requests: int = 10) -> Dict[str, Any]:
        """Test API performance with multiple requests."""
        print(f"Testing performance with {num_requests} requests...")
        
        image_bytes = self.create_test_image()
        files = {"file": ("test.jpg", image_bytes, "image/jpeg")}
        
        response_times = []
        success_count = 0
        
        for i in range(num_requests):
            result = self.test_endpoint("/detect", "POST", files=files)
            response_times.append(result["response_time"])
            
            if result["success"]:
                success_count += 1
            
            if (i + 1) % 5 == 0:
                print(f"  Completed {i + 1}/{num_requests} requests")
        
        # Calculate statistics
        avg_time = sum(response_times) / len(response_times)
        min_time = min(response_times)
        max_time = max(response_times)
        success_rate = (success_count / num_requests) * 100
        
        performance_result = {
            "num_requests": num_requests,
            "success_count": success_count,
            "success_rate": success_rate,
            "avg_response_time": avg_time,
            "min_response_time": min_time,
            "max_response_time": max_time,
            "total_time": sum(response_times)
        }
        
        print(f"  Success rate: {success_rate:.1f}%")
        print(f"  Average response time: {avg_time:.3f}s")
        print(f"  Min/Max response time: {min_time:.3f}s / {max_time:.3f}s")
        
        return performance_result
    
    def test_statistics_endpoints(self) -> List[Dict[str, Any]]:
        """Test statistics and monitoring endpoints."""
        print("Testing statistics endpoints...")
        
        results = []
        
        # Test stats endpoint
        result = self.test_endpoint("/stats", "GET")
        results.append(result)
        
        status = "✓" if result["success"] else "✗"
        print(f"  {status} GET /stats: {result['status_code']}")
        
        # Test stats reset
        result = self.test_endpoint("/stats/reset", "POST")
        results.append(result)
        
        status = "✓" if result["success"] else "✗"
        print(f"  {status} POST /stats/reset: {result['status_code']}")
        
        return results
    
    def run_all_tests(self, performance_requests: int = 10) -> Dict[str, Any]:
        """Run all tests and return comprehensive results."""
        print(f"Starting comprehensive API tests for: {self.base_url}")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run all test suites
        basic_results = self.test_basic_endpoints()
        single_result = self.test_single_image_detection()
        batch_result = self.test_batch_image_detection()
        error_results = self.test_error_conditions()
        performance_result = self.test_performance(performance_requests)
        stats_results = self.test_statistics_endpoints()
        
        total_time = time.time() - start_time
        
        # Compile results
        all_results = basic_results + [single_result, batch_result] + error_results + stats_results
        
        success_count = sum(1 for r in all_results if r["success"])
        total_tests = len(all_results)
        
        summary = {
            "total_tests": total_tests,
            "successful_tests": success_count,
            "failed_tests": total_tests - success_count,
            "success_rate": (success_count / total_tests) * 100,
            "total_time": total_time,
            "performance": performance_result,
            "detailed_results": all_results
        }
        
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Total tests: {total_tests}")
        print(f"Successful: {success_count}")
        print(f"Failed: {total_tests - success_count}")
        print(f"Success rate: {summary['success_rate']:.1f}%")
        print(f"Total time: {total_time:.3f}s")
        
        if summary['success_rate'] < 80:
            print("\n⚠️  WARNING: Success rate is below 80%")
        elif summary['success_rate'] == 100:
            print("\n🎉 All tests passed!")
        
        return summary

def main():
    parser = argparse.ArgumentParser(description="Test Image Recognition API")
    parser.add_argument("--url", default="http://localhost:8080", 
                       help="Base URL of the API (default: http://localhost:8080)")
    parser.add_argument("--performance", type=int, default=10,
                       help="Number of requests for performance testing (default: 10)")
    parser.add_argument("--output", help="Output file for detailed results (JSON)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Create tester and run tests
    tester = APITester(args.url)
    results = tester.run_all_tests(args.performance)
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {args.output}")
    
    # Exit with appropriate code
    sys.exit(0 if results['success_rate'] == 100 else 1)

if __name__ == "__main__":
    main() 
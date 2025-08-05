#!/usr/bin/env python3
"""
Test script for Text Translation API
"""

import requests
import json
import time
import argparse
from typing import Dict, Any


def test_health(base_url: str) -> bool:
    """Test health endpoint."""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=30)
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ Health check: {data['status']}")
        print(f"   Model loaded: {data['model_loaded']}")
        print(f"   Model healthy: {data['model_healthy']}")
        
        return data['status'] == 'healthy'
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


def test_single_translation(base_url: str) -> bool:
    """Test single translation."""
    print("\n🔍 Testing single translation...")
    
    test_cases = [
        {
            "text": "Hello, world!",
            "source_language": "english",
            "target_language": "spanish"
        },
        {
            "text": "How are you today?",
            "source_language": "en",
            "target_language": "fr"
        },
        {
            "text": "Good morning",
            "source_language": "eng_Latn",
            "target_language": "deu_Latn"
        }
    ]
    
    success = True
    
    for i, test_case in enumerate(test_cases, 1):
        try:
            print(f"   Test {i}: {test_case['text']} ({test_case['source_language']} → {test_case['target_language']})")
            
            start_time = time.time()
            response = requests.post(
                f"{base_url}/translate",
                json=test_case,
                timeout=60
            )
            response.raise_for_status()
            end_time = time.time()
            
            data = response.json()
            
            print(f"   ✅ Translation: '{data['translated_text']}'")
            print(f"   ⏱️  Time: {end_time - start_time:.2f}s (inference: {data['inference_time']:.2f}s)")
            
        except Exception as e:
            print(f"   ❌ Test {i} failed: {e}")
            success = False
    
    return success


def test_batch_translation(base_url: str) -> bool:
    """Test batch translation."""
    print("\n🔍 Testing batch translation...")
    
    test_case = {
        "texts": [
            "Hello",
            "How are you?",
            "Good morning",
            "Thank you",
            "Goodbye"
        ],
        "source_language": "english",
        "target_language": "spanish"
    }
    
    try:
        print(f"   Translating {len(test_case['texts'])} texts...")
        
        start_time = time.time()
        response = requests.post(
            f"{base_url}/translate/batch",
            json=test_case,
            timeout=120
        )
        response.raise_for_status()
        end_time = time.time()
        
        data = response.json()
        
        print(f"   ✅ Batch translation completed")
        print(f"   📊 Results: {data['batch_size']} translations")
        print(f"   ⏱️  Total time: {data['total_time']:.2f}s")
        print(f"   ⏱️  Average time: {data['average_time_per_translation']:.2f}s per translation")
        
        for i, translation in enumerate(data['translations']):
            print(f"      {i+1}. '{translation['original_text']}' → '{translation['translated_text']}'")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Batch translation failed: {e}")
        return False


def test_supported_languages(base_url: str) -> bool:
    """Test supported languages endpoint."""
    print("\n🔍 Testing supported languages...")
    
    try:
        response = requests.get(f"{base_url}/languages", timeout=30)
        response.raise_for_status()
        data = response.json()
        
        print(f"   ✅ Supported languages: {data['total_count']}")
        print("   📝 Sample languages:")
        
        # Show first 10 languages
        for i, (name, code) in enumerate(list(data['languages'].items())[:10]):
            print(f"      {name} → {code}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Languages test failed: {e}")
        return False


def test_model_info(base_url: str) -> bool:
    """Test model info endpoint."""
    print("\n🔍 Testing model info...")
    
    try:
        response = requests.get(f"{base_url}/model/info", timeout=30)
        response.raise_for_status()
        data = response.json()
        
        print(f"   ✅ Model: {data['model_name']}")
        print(f"   🖥️  Device: {data['device']}")
        print(f"   ⚡ Quantization: {data['quantization_type'] if data['quantization_enabled'] else 'None'}")
        print(f"   📊 Supported languages: {data['supported_languages']}")
        print(f"   📈 Total requests: {data['stats']['total_requests']}")
        print(f"   ⏱️  Average inference time: {data['stats']['average_inference_time']:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Model info test failed: {e}")
        return False


def test_error_handling(base_url: str) -> bool:
    """Test error handling."""
    print("\n🔍 Testing error handling...")
    
    error_tests = [
        {
            "name": "Invalid source language",
            "data": {
                "text": "Hello",
                "source_language": "invalid_lang",
                "target_language": "spanish"
            },
            "expected_status": 400
        },
        {
            "name": "Empty text",
            "data": {
                "text": "",
                "source_language": "english",
                "target_language": "spanish"
            },
            "expected_status": 422
        },
        {
            "name": "Very long text",
            "data": {
                "text": "x" * 10000,  # Very long text
                "source_language": "english",
                "target_language": "spanish"
            },
            "expected_status": 400
        }
    ]
    
    success = True
    
    for test in error_tests:
        try:
            print(f"   Testing: {test['name']}")
            
            response = requests.post(
                f"{base_url}/translate",
                json=test['data'],
                timeout=30
            )
            
            if response.status_code == test['expected_status']:
                print(f"   ✅ Correctly returned status {response.status_code}")
            else:
                print(f"   ❌ Expected status {test['expected_status']}, got {response.status_code}")
                success = False
                
        except Exception as e:
            print(f"   ❌ Error test failed: {e}")
            success = False
    
    return success


def run_performance_test(base_url: str, num_requests: int = 10) -> bool:
    """Run performance test."""
    print(f"\n🚀 Running performance test ({num_requests} requests)...")
    
    test_data = {
        "text": "This is a performance test for the translation API.",
        "source_language": "english",
        "target_language": "spanish"
    }
    
    times = []
    successful = 0
    
    for i in range(num_requests):
        try:
            start_time = time.time()
            response = requests.post(
                f"{base_url}/translate",
                json=test_data,
                timeout=60
            )
            response.raise_for_status()
            end_time = time.time()
            
            times.append(end_time - start_time)
            successful += 1
            
            if i % 5 == 0:
                print(f"   Progress: {i+1}/{num_requests}")
                
        except Exception as e:
            print(f"   ❌ Request {i+1} failed: {e}")
    
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"   ✅ Performance results:")
        print(f"      Successful requests: {successful}/{num_requests}")
        print(f"      Average time: {avg_time:.2f}s")
        print(f"      Min time: {min_time:.2f}s")
        print(f"      Max time: {max_time:.2f}s")
        
        return successful > 0
    else:
        print(f"   ❌ No successful requests")
        return False


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test Text Translation API")
    parser.add_argument("base_url", help="Base URL of the API (e.g., https://your-service-url)")
    parser.add_argument("--performance", "-p", action="store_true", help="Run performance test")
    parser.add_argument("--num-requests", "-n", type=int, default=10, help="Number of requests for performance test")
    
    args = parser.parse_args()
    base_url = args.base_url.rstrip('/')
    
    print(f"🌍 Testing Text Translation API at: {base_url}")
    print("=" * 60)
    
    # Run tests
    tests = [
        ("Health Check", lambda: test_health(base_url)),
        ("Single Translation", lambda: test_single_translation(base_url)),
        ("Batch Translation", lambda: test_batch_translation(base_url)),
        ("Supported Languages", lambda: test_supported_languages(base_url)),
        ("Model Info", lambda: test_model_info(base_url)),
        ("Error Handling", lambda: test_error_handling(base_url)),
    ]
    
    if args.performance:
        tests.append(("Performance Test", lambda: run_performance_test(base_url, args.num_requests)))
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"🧪 {test_name}")
        print('='*60)
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print('='*60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed")
        return 1


if __name__ == "__main__":
    exit(main())
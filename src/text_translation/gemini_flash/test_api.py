#!/usr/bin/env python3
"""
Test script for Gemini Flash Translation API
"""
import asyncio
import json
import time
import sys
from typing import Dict, Any, List
import httpx


class GeminiFlashAPITester:
    """Test client for Gemini Flash Translation API"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def test_health(self) -> Dict[str, Any]:
        """Test health endpoint"""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            response.raise_for_status()
            return {"status": "success", "data": response.json()}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def test_translation(self, text: str = "Hello, world!", 
                             source_lang: str = "english", 
                             target_lang: str = "spanish") -> Dict[str, Any]:
        """Test single translation"""
        try:
            payload = {
                "text": text,
                "source_language": source_lang,
                "target_language": target_lang
            }
            
            start_time = time.time()
            response = await self.client.post(
                f"{self.base_url}/translate",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            end_time = time.time()
            
            response.raise_for_status()
            result = response.json()
            result["api_response_time"] = end_time - start_time
            
            return {"status": "success", "data": result}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def test_batch_translation(self, texts: List[str] = None,
                                   source_lang: str = "english",
                                   target_lang: str = "spanish") -> Dict[str, Any]:
        """Test batch translation"""
        if texts is None:
            texts = ["Hello", "World", "How are you?", "Good morning", "Thank you"]
        
        try:
            payload = {
                "texts": texts,
                "source_language": source_lang,
                "target_language": target_lang
            }
            
            start_time = time.time()
            response = await self.client.post(
                f"{self.base_url}/translate/batch",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            end_time = time.time()
            
            response.raise_for_status()
            result = response.json()
            result["api_response_time"] = end_time - start_time
            
            return {"status": "success", "data": result}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def test_languages(self) -> Dict[str, Any]:
        """Test supported languages endpoint"""
        try:
            response = await self.client.get(f"{self.base_url}/languages")
            response.raise_for_status()
            return {"status": "success", "data": response.json()}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def test_model_info(self) -> Dict[str, Any]:
        """Test model info endpoint"""
        try:
            response = await self.client.get(f"{self.base_url}/model/info")
            response.raise_for_status()
            return {"status": "success", "data": response.json()}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def test_stats(self) -> Dict[str, Any]:
        """Test statistics endpoint"""
        try:
            response = await self.client.get(f"{self.base_url}/stats")
            response.raise_for_status()
            return {"status": "success", "data": response.json()}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def test_cache_info(self) -> Dict[str, Any]:
        """Test cache info endpoint"""
        try:
            response = await self.client.get(f"{self.base_url}/cache/info")
            response.raise_for_status()
            return {"status": "success", "data": response.json()}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def performance_test(self, num_requests: int = 5) -> Dict[str, Any]:
        """Run performance test with multiple requests"""
        print(f"\n🚀 Running performance test with {num_requests} requests...")
        
        tasks = []
        test_texts = [
            "Hello, how are you today?",
            "The weather is beautiful.",
            "I love programming.",
            "Artificial intelligence is amazing.",
            "Good morning, everyone!"
        ]
        
        for i in range(num_requests):
            text = test_texts[i % len(test_texts)]
            tasks.append(self.test_translation(text))
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        successful = [r for r in results if r["status"] == "success"]
        failed = [r for r in results if r["status"] == "error"]
        
        if successful:
            inference_times = [r["data"]["inference_time"] for r in successful]
            api_times = [r["data"]["api_response_time"] for r in successful]
            
            performance_stats = {
                "total_requests": num_requests,
                "successful_requests": len(successful),
                "failed_requests": len(failed),
                "total_time": end_time - start_time,
                "avg_inference_time": sum(inference_times) / len(inference_times),
                "min_inference_time": min(inference_times),
                "max_inference_time": max(inference_times),
                "avg_api_response_time": sum(api_times) / len(api_times),
                "requests_per_second": num_requests / (end_time - start_time)
            }
        else:
            performance_stats = {
                "total_requests": num_requests,
                "successful_requests": 0,
                "failed_requests": len(failed),
                "error": "All requests failed"
            }
        
        return {
            "status": "success" if successful else "error",
            "performance": performance_stats,
            "results": results
        }
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()


def print_result(title: str, result: Dict[str, Any]):
    """Print test result in a formatted way"""
    print(f"\n{'='*50}")
    print(f"🧪 {title}")
    print(f"{'='*50}")
    
    if result["status"] == "success":
        print("✅ SUCCESS")
        if "data" in result:
            data = result["data"]
            if isinstance(data, dict):
                for key, value in data.items():
                    if key == "inference_time":
                        print(f"⏱️  Inference Time: {value:.3f}s")
                    elif key == "api_response_time":
                        print(f"🌐 API Response Time: {value:.3f}s")
                    elif key == "translated_text":
                        print(f"📝 Translation: {value}")
                    elif key == "total_languages":
                        print(f"🌍 Supported Languages: {value}")
                    elif key == "model_name":
                        print(f"🤖 Model: {value}")
                    elif key == "cache_hit_rate":
                        print(f"💾 Cache Hit Rate: {value:.1f}%")
                    elif isinstance(value, (str, int, float, bool)):
                        print(f"   {key}: {value}")
            else:
                print(f"   Data: {data}")
    else:
        print("❌ FAILED")
        print(f"   Error: {result['error']}")


async def main():
    """Main test function"""
    if len(sys.argv) != 2:
        print("Usage: python test_api.py <API_URL>")
        print("Example: python test_api.py https://gemini-flash-api-xyz.run.app")
        sys.exit(1)
    
    base_url = sys.argv[1]
    print(f"🧪 Testing Gemini Flash Translation API at: {base_url}")
    
    tester = GeminiFlashAPITester(base_url)
    
    try:
        # Test 1: Health Check
        health_result = await tester.test_health()
        print_result("Health Check", health_result)
        
        if health_result["status"] != "success":
            print("\n❌ Health check failed. Service might not be ready.")
            return
        
        # Test 2: Model Info
        model_result = await tester.test_model_info()
        print_result("Model Information", model_result)
        
        # Test 3: Supported Languages
        languages_result = await tester.test_languages()
        print_result("Supported Languages", languages_result)
        
        # Test 4: Single Translation
        translation_result = await tester.test_translation()
        print_result("Single Translation", translation_result)
        
        # Test 5: Batch Translation
        batch_result = await tester.test_batch_translation()
        print_result("Batch Translation", batch_result)
        
        # Test 6: Cache and Stats
        stats_result = await tester.test_stats()
        print_result("Statistics", stats_result)
        
        cache_result = await tester.test_cache_info()
        print_result("Cache Information", cache_result)
        
        # Test 7: Performance Test
        if translation_result["status"] == "success":
            perf_result = await tester.performance_test(5)
            print_result("Performance Test", perf_result)
            
            if perf_result["status"] == "success":
                perf = perf_result["performance"]
                print(f"\n🏆 Performance Summary:")
                print(f"   Average Inference Time: {perf['avg_inference_time']:.3f}s")
                print(f"   Requests per Second: {perf['requests_per_second']:.2f}")
                print(f"   Success Rate: {perf['successful_requests']}/{perf['total_requests']}")
        
        # Test 8: Different Language Pairs
        print(f"\n{'='*50}")
        print("🌍 Testing Different Language Pairs")
        print(f"{'='*50}")
        
        language_pairs = [
            ("english", "french", "Good morning!"),
            ("spanish", "english", "¡Hola mundo!"),
            ("english", "german", "How are you?"),
            ("french", "spanish", "Bonjour le monde"),
        ]
        
        for source, target, text in language_pairs:
            result = await tester.test_translation(text, source, target)
            if result["status"] == "success":
                data = result["data"]
                print(f"✅ {source} → {target}: '{text}' → '{data['translated_text']}' ({data['inference_time']:.3f}s)")
            else:
                print(f"❌ {source} → {target}: Failed - {result['error']}")
        
        print(f"\n🎉 All tests completed!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    finally:
        await tester.close()


if __name__ == "__main__":
    asyncio.run(main())
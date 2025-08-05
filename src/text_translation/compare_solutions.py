#!/usr/bin/env python3
"""
Compare NLLB and Gemini Flash translation solutions
"""
import asyncio
import json
import time
import sys
from typing import Dict, Any, List
import httpx


class TranslationComparison:
    """Compare different translation solutions"""
    
    def __init__(self, nllb_url: str = None, gemini_url: str = None):
        self.nllb_url = nllb_url.rstrip('/') if nllb_url else None
        self.gemini_url = gemini_url.rstrip('/') if gemini_url else None
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def test_nllb(self, text: str, source_lang: str, target_lang: str) -> Dict[str, Any]:
        """Test NLLB translation"""
        if not self.nllb_url:
            return {"status": "error", "error": "NLLB URL not provided"}
        
        try:
            payload = {
                "text": text,
                "source_language": source_lang,
                "target_language": target_lang
            }
            
            start_time = time.time()
            response = await self.client.post(
                f"{self.nllb_url}/translate",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            end_time = time.time()
            
            response.raise_for_status()
            result = response.json()
            
            return {
                "status": "success",
                "solution": "NLLB",
                "translated_text": result.get("translated_text", ""),
                "inference_time": result.get("inference_time", 0),
                "api_response_time": end_time - start_time,
                "model": "NLLB-200-distilled-600M",
                "cached": result.get("cached", False)
            }
        except Exception as e:
            return {
                "status": "error",
                "solution": "NLLB",
                "error": str(e)
            }
    
    async def test_gemini(self, text: str, source_lang: str, target_lang: str) -> Dict[str, Any]:
        """Test Gemini Flash translation"""
        if not self.gemini_url:
            return {"status": "error", "error": "Gemini URL not provided"}
        
        try:
            payload = {
                "text": text,
                "source_language": source_lang,
                "target_language": target_lang
            }
            
            start_time = time.time()
            response = await self.client.post(
                f"{self.gemini_url}/translate",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            end_time = time.time()
            
            response.raise_for_status()
            result = response.json()
            
            return {
                "status": "success",
                "solution": "Gemini Flash",
                "translated_text": result.get("translated_text", ""),
                "inference_time": result.get("inference_time", 0),
                "api_response_time": end_time - start_time,
                "model": result.get("model", "gemini-1.5-flash"),
                "cached": result.get("cached", False)
            }
        except Exception as e:
            return {
                "status": "error",
                "solution": "Gemini Flash",
                "error": str(e)
            }
    
    async def compare_translation(self, text: str, source_lang: str, target_lang: str) -> Dict[str, Any]:
        """Compare both solutions for a single translation"""
        print(f"\n🔄 Testing: '{text}' ({source_lang} → {target_lang})")
        print("-" * 60)
        
        # Run both tests concurrently
        tasks = []
        if self.nllb_url:
            tasks.append(self.test_nllb(text, source_lang, target_lang))
        if self.gemini_url:
            tasks.append(self.test_gemini(text, source_lang, target_lang))
        
        if not tasks:
            return {"error": "No URLs provided"}
        
        results = await asyncio.gather(*tasks)
        
        # Display results
        comparison = {"original_text": text, "source_lang": source_lang, "target_lang": target_lang, "results": {}}
        
        for result in results:
            solution = result.get("solution", "Unknown")
            comparison["results"][solution] = result
            
            if result["status"] == "success":
                print(f"✅ {solution}:")
                print(f"   Translation: {result['translated_text']}")
                print(f"   Inference Time: {result['inference_time']:.3f}s")
                print(f"   API Response Time: {result['api_response_time']:.3f}s")
                print(f"   Model: {result['model']}")
                if result.get('cached'):
                    print(f"   💾 Cached result")
            else:
                print(f"❌ {solution}: {result['error']}")
        
        return comparison
    
    async def performance_comparison(self, num_tests: int = 5) -> Dict[str, Any]:
        """Run performance comparison"""
        print(f"\n🏁 Performance Comparison ({num_tests} tests each)")
        print("=" * 60)
        
        test_cases = [
            ("Hello, how are you today?", "english", "spanish"),
            ("The weather is beautiful outside.", "english", "french"),
            ("I love learning new languages.", "english", "german"),
            ("Technology is advancing rapidly.", "english", "italian"),
            ("Good morning, have a great day!", "english", "portuguese")
        ]
        
        all_results = []
        
        for i in range(num_tests):
            text, source, target = test_cases[i % len(test_cases)]
            result = await self.compare_translation(text, source, target)
            all_results.append(result)
            
            # Small delay between tests
            await asyncio.sleep(0.5)
        
        # Calculate statistics
        stats = {"NLLB": [], "Gemini Flash": []}
        
        for result in all_results:
            for solution, data in result.get("results", {}).items():
                if data["status"] == "success":
                    stats[solution].append({
                        "inference_time": data["inference_time"],
                        "api_response_time": data["api_response_time"]
                    })
        
        # Generate summary
        summary = {}
        for solution, times in stats.items():
            if times:
                inference_times = [t["inference_time"] for t in times]
                api_times = [t["api_response_time"] for t in times]
                
                summary[solution] = {
                    "total_tests": len(times),
                    "avg_inference_time": sum(inference_times) / len(inference_times),
                    "min_inference_time": min(inference_times),
                    "max_inference_time": max(inference_times),
                    "avg_api_response_time": sum(api_times) / len(api_times),
                    "success_rate": len(times) / num_tests * 100
                }
            else:
                summary[solution] = {
                    "total_tests": 0,
                    "success_rate": 0,
                    "error": "No successful tests"
                }
        
        return {
            "test_count": num_tests,
            "detailed_results": all_results,
            "performance_summary": summary
        }
    
    def print_summary(self, performance_data: Dict[str, Any]):
        """Print performance summary"""
        print(f"\n📊 Performance Summary")
        print("=" * 60)
        
        summary = performance_data["performance_summary"]
        
        for solution, stats in summary.items():
            print(f"\n🚀 {solution}:")
            if "error" not in stats:
                print(f"   Success Rate: {stats['success_rate']:.1f}%")
                print(f"   Avg Inference Time: {stats['avg_inference_time']:.3f}s")
                print(f"   Min/Max Inference: {stats['min_inference_time']:.3f}s / {stats['max_inference_time']:.3f}s")
                print(f"   Avg API Response: {stats['avg_api_response_time']:.3f}s")
                
                # Performance rating
                avg_time = stats['avg_inference_time']
                if avg_time < 0.5:
                    rating = "⚡ Excellent"
                elif avg_time < 1.0:
                    rating = "🚀 Very Good"
                elif avg_time < 2.0:
                    rating = "✅ Good"
                else:
                    rating = "🐌 Slow"
                
                print(f"   Performance: {rating}")
            else:
                print(f"   ❌ {stats['error']}")
        
        # Winner determination
        if len(summary) == 2:
            solutions = list(summary.keys())
            sol1, sol2 = solutions[0], solutions[1]
            
            if "error" not in summary[sol1] and "error" not in summary[sol2]:
                time1 = summary[sol1]['avg_inference_time']
                time2 = summary[sol2]['avg_inference_time']
                
                if time1 < time2:
                    winner = sol1
                    improvement = (time2 - time1) / time2 * 100
                else:
                    winner = sol2
                    improvement = (time1 - time2) / time1 * 100
                
                print(f"\n🏆 Winner: {winner}")
                print(f"   {improvement:.1f}% faster on average")
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()


async def main():
    """Main comparison function"""
    if len(sys.argv) < 2:
        print("Usage: python compare_solutions.py [--nllb NLLB_URL] [--gemini GEMINI_URL]")
        print("\nExample:")
        print("python compare_solutions.py \\")
        print("  --nllb https://text-translation-api-xyz.run.app \\")
        print("  --gemini https://gemini-flash-api-abc.run.app")
        sys.exit(1)
    
    nllb_url = None
    gemini_url = None
    
    # Parse arguments
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--nllb" and i + 1 < len(sys.argv):
            nllb_url = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--gemini" and i + 1 < len(sys.argv):
            gemini_url = sys.argv[i + 1]
            i += 2
        else:
            i += 1
    
    if not nllb_url and not gemini_url:
        print("❌ Please provide at least one URL (--nllb or --gemini)")
        sys.exit(1)
    
    print("🔍 Translation Solutions Comparison")
    print("=" * 50)
    if nllb_url:
        print(f"NLLB URL: {nllb_url}")
    if gemini_url:
        print(f"Gemini URL: {gemini_url}")
    
    comparator = TranslationComparison(nllb_url, gemini_url)
    
    try:
        # Single comparison tests
        test_cases = [
            ("Hello, world!", "english", "spanish"),
            ("How are you doing today?", "english", "french"),
            ("The weather is beautiful.", "english", "german"),
        ]
        
        for text, source, target in test_cases:
            await comparator.compare_translation(text, source, target)
        
        # Performance comparison
        performance_data = await comparator.performance_comparison(5)
        comparator.print_summary(performance_data)
        
        print(f"\n✅ Comparison completed!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Comparison interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    finally:
        await comparator.close()


if __name__ == "__main__":
    asyncio.run(main())
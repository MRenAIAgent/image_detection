"""
Gemini Flash Translation Client
"""
import asyncio
import time
import logging
from typing import List, Optional, Dict, Any
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from .config import settings

logger = logging.getLogger(__name__)


class GeminiTranslationClient:
    """Client for Gemini Flash translation"""
    
    def __init__(self):
        self.model = None
        self.stats = {
            "total_requests": 0,
            "total_characters": 0,
            "total_inference_time": 0.0,
            "errors": 0,
            "cache_hits": 0
        }
        self._cache = {}  # Simple in-memory cache
        self._max_cache_size = 1000
        
    async def initialize(self):
        """Initialize the Gemini client"""
        try:
            if not settings.google_api_key:
                raise ValueError("GOOGLE_API_KEY environment variable is required")
            
            # Configure the API key
            genai.configure(api_key=settings.google_api_key)
            
            # Initialize the model with safety settings
            generation_config = {
                "temperature": settings.default_temperature,
                "top_p": 0.95,
                "top_k": 64,
                "max_output_tokens": 8192,
            }
            
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
            
            self.model = genai.GenerativeModel(
                model_name=settings.gemini_model,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            logger.info(f"Gemini client initialized with model: {settings.gemini_model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise
    
    def _normalize_language(self, language: str) -> str:
        """Normalize language code to full name"""
        language = language.lower().strip()
        return settings.language_mappings.get(language, language)
    
    def _get_cache_key(self, text: str, source_lang: str, target_lang: str) -> str:
        """Generate cache key for translation"""
        return f"{source_lang}:{target_lang}:{hash(text)}"
    
    def _create_translation_prompt(self, text: str, source_lang: str, target_lang: str) -> str:
        """Create optimized prompt for translation"""
        source_lang = self._normalize_language(source_lang)
        target_lang = self._normalize_language(target_lang)
        
        # Optimized prompt for better translation quality
        prompt = f"""Translate the following {source_lang} text to {target_lang}. 
Provide only the translation without any explanations, comments, or additional text.

Text to translate: {text}

Translation:"""
        
        return prompt
    
    async def translate(
        self, 
        text: str, 
        source_language: str, 
        target_language: str,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """Translate text using Gemini Flash"""
        start_time = time.time()
        
        try:
            # Validate inputs
            if not text or not text.strip():
                raise ValueError("Text cannot be empty")
            
            if len(text) > settings.max_text_length:
                raise ValueError(f"Text too long. Maximum {settings.max_text_length} characters allowed")
            
            # Normalize languages
            source_lang = self._normalize_language(source_language)
            target_lang = self._normalize_language(target_language)
            
            # Check cache
            cache_key = self._get_cache_key(text, source_lang, target_lang)
            if use_cache and cache_key in self._cache:
                self.stats["cache_hits"] += 1
                cached_result = self._cache[cache_key].copy()
                cached_result["inference_time"] = time.time() - start_time
                cached_result["cached"] = True
                return cached_result
            
            # Create prompt
            prompt = self._create_translation_prompt(text, source_lang, target_lang)
            
            # Generate translation
            response = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.model.generate_content(prompt)
            )
            
            if not response or not response.text:
                raise ValueError("Empty response from Gemini")
            
            translated_text = response.text.strip()
            
            # Remove any unwanted prefixes/suffixes
            if translated_text.startswith("Translation:"):
                translated_text = translated_text[12:].strip()
            
            inference_time = time.time() - start_time
            
            result = {
                "translated_text": translated_text,
                "source_language": source_lang,
                "target_language": target_lang,
                "original_text": text,
                "inference_time": inference_time,
                "model": settings.gemini_model,
                "cached": False
            }
            
            # Update cache (with size limit)
            if use_cache:
                if len(self._cache) >= self._max_cache_size:
                    # Remove oldest entry
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
                self._cache[cache_key] = result.copy()
            
            # Update stats
            self.stats["total_requests"] += 1
            self.stats["total_characters"] += len(text)
            self.stats["total_inference_time"] += inference_time
            
            return result
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Translation error: {e}")
            raise
    
    async def translate_batch(
        self, 
        texts: List[str], 
        source_language: str, 
        target_language: str,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """Translate multiple texts in batch"""
        if len(texts) > settings.max_batch_size:
            raise ValueError(f"Batch size too large. Maximum {settings.max_batch_size} texts allowed")
        
        # Process translations concurrently
        tasks = [
            self.translate(text, source_language, target_language, use_cache)
            for text in texts
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "error": str(result),
                    "original_text": texts[i],
                    "source_language": source_language,
                    "target_language": target_language
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get translation statistics"""
        stats = self.stats.copy()
        if stats["total_requests"] > 0:
            stats["avg_inference_time"] = stats["total_inference_time"] / stats["total_requests"]
            stats["avg_characters_per_request"] = stats["total_characters"] / stats["total_requests"]
        else:
            stats["avg_inference_time"] = 0.0
            stats["avg_characters_per_request"] = 0.0
        
        stats["cache_size"] = len(self._cache)
        stats["cache_hit_rate"] = (
            stats["cache_hits"] / max(stats["total_requests"], 1) * 100
        )
        
        return stats
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            "total_requests": 0,
            "total_characters": 0,
            "total_inference_time": 0.0,
            "errors": 0,
            "cache_hits": 0
        }
    
    def clear_cache(self):
        """Clear translation cache"""
        self._cache.clear()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_name": settings.gemini_model,
            "provider": "Google",
            "type": "Generative AI",
            "max_text_length": settings.max_text_length,
            "max_batch_size": settings.max_batch_size,
            "supported_languages": len(settings.supported_languages),
            "cache_enabled": True,
            "cache_size": len(self._cache),
            "max_cache_size": self._max_cache_size
        }


# Global client instance
translation_client = GeminiTranslationClient()
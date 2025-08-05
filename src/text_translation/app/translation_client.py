"""
NLLB Translation Client with quantization support
"""

import asyncio
import logging
import time
import torch
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import psutil
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig
)

from .config import settings

logger = logging.getLogger(__name__)


class TranslationResult:
    """Translation result container."""
    
    def __init__(self, translated_text: str, source_lang: str, target_lang: str, 
                 inference_time: float, confidence: Optional[float] = None):
        self.translated_text = translated_text
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.inference_time = inference_time
        self.confidence = confidence


class NLLBTranslationClient:
    """NLLB Translation client with quantization support."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.model_loaded = False
        self.stats = {
            "total_requests": 0,
            "total_inference_time": 0.0,
            "average_inference_time": 0.0,
            "cache_hits": 0,
            "errors": 0,
            "model_load_time": 0.0
        }
        self._cache = {} if settings.enable_cache else None
        self._lock = asyncio.Lock()
    
    async def initialize(self) -> bool:
        """Initialize the NLLB model and tokenizer."""
        try:
            start_time = time.time()
            logger.info(f"Initializing NLLB model: {settings.model_name}")
            
            # Determine device
            if settings.device == "auto":
                if torch.cuda.is_available():
                    self.device = "cuda"
                    logger.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name()}")
                else:
                    self.device = "cpu"
                    logger.info("CUDA not available. Using CPU")
            else:
                self.device = settings.device
            
            # Create model cache directory
            cache_dir = Path(settings.model_cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.model_name,
                cache_dir=cache_dir,
                use_fast=True
            )
            
            # Configure quantization
            model_kwargs = {
                "cache_dir": cache_dir,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            }
            
            if settings.quantization_enabled:
                logger.info(f"Enabling quantization: {settings.quantization_type}")
                
                if settings.quantization_type == "int8" and self.device == "cuda":
                    # Use 8-bit quantization for CUDA
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_enable_fp32_cpu_offload=True
                    )
                    model_kwargs["quantization_config"] = quantization_config
                    model_kwargs["device_map"] = "auto"
                    
                elif settings.quantization_type == "int4" and self.device == "cuda":
                    # Use 4-bit quantization for CUDA
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    model_kwargs["quantization_config"] = quantization_config
                    model_kwargs["device_map"] = "auto"
                    
                elif settings.quantization_type == "fp16":
                    # Use half precision
                    model_kwargs["torch_dtype"] = torch.float16
            
            # Load model
            logger.info("Loading NLLB model...")
            memory_before = psutil.virtual_memory().used / 1024**3
            
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                settings.model_name,
                **model_kwargs
            )
            
            # Move to device if not using device_map
            if "device_map" not in model_kwargs:
                self.model = self.model.to(self.device)
            
            memory_after = psutil.virtual_memory().used / 1024**3
            memory_used = memory_after - memory_before
            
            # Enable optimizations
            if settings.use_bettertransformer and hasattr(self.model, 'to_bettertransformer'):
                try:
                    logger.info("Enabling BetterTransformer optimization...")
                    self.model = self.model.to_bettertransformer()
                except Exception as e:
                    logger.warning(f"BetterTransformer not available: {e}")
            
            if settings.torch_compile and hasattr(torch, 'compile'):
                try:
                    logger.info("Enabling torch.compile optimization...")
                    self.model = torch.compile(self.model)
                except Exception as e:
                    logger.warning(f"torch.compile not available: {e}")
            
            # Set to evaluation mode
            self.model.eval()
            
            load_time = time.time() - start_time
            self.stats["model_load_time"] = load_time
            self.model_loaded = True
            
            logger.info(f"Model loaded successfully in {load_time:.2f}s")
            logger.info(f"Memory used: {memory_used:.2f} GB")
            logger.info(f"Device: {self.device}")
            logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            return False
    
    async def translate(self, text: str, source_lang: str, target_lang: str) -> TranslationResult:
        """Translate text from source to target language."""
        if not self.model_loaded:
            raise RuntimeError("Model not initialized")
        
        start_time = time.time()
        
        try:
            # Check cache
            cache_key = f"{text}_{source_lang}_{target_lang}"
            if self._cache and cache_key in self._cache:
                cached_result = self._cache[cache_key]
                self.stats["cache_hits"] += 1
                logger.debug("Cache hit for translation request")
                return cached_result
            
            async with self._lock:
                # Tokenize input
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=settings.max_length
                ).to(self.device)
                
                # Set target language token
                target_lang_token = self.tokenizer.lang_code_to_id.get(target_lang)
                if target_lang_token is None:
                    raise ValueError(f"Target language {target_lang} not supported")
                
                # Generate translation
                with torch.no_grad():
                    generated_tokens = self.model.generate(
                        **inputs,
                        forced_bos_token_id=target_lang_token,
                        max_length=settings.max_length,
                        num_beams=4,
                        length_penalty=1.0,
                        early_stopping=True,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                
                # Decode result
                translated_text = self.tokenizer.batch_decode(
                    generated_tokens, 
                    skip_special_tokens=True
                )[0]
                
                inference_time = time.time() - start_time
                
                # Update stats
                self.stats["total_requests"] += 1
                self.stats["total_inference_time"] += inference_time
                self.stats["average_inference_time"] = (
                    self.stats["total_inference_time"] / self.stats["total_requests"]
                )
                
                # Create result
                result = TranslationResult(
                    translated_text=translated_text,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    inference_time=inference_time
                )
                
                # Cache result
                if self._cache and len(self._cache) < 1000:  # Limit cache size
                    self._cache[cache_key] = result
                
                logger.debug(f"Translation completed in {inference_time:.3f}s")
                return result
                
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Translation error: {e}")
            raise
    
    async def translate_batch(self, texts: List[str], source_lang: str, 
                            target_lang: str) -> List[TranslationResult]:
        """Translate multiple texts in batch."""
        if not texts:
            return []
        
        if len(texts) == 1:
            return [await self.translate(texts[0], source_lang, target_lang)]
        
        start_time = time.time()
        
        try:
            async with self._lock:
                # Tokenize all inputs
                inputs = self.tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=settings.max_length
                ).to(self.device)
                
                # Set target language token
                target_lang_token = self.tokenizer.lang_code_to_id.get(target_lang)
                if target_lang_token is None:
                    raise ValueError(f"Target language {target_lang} not supported")
                
                # Generate translations
                with torch.no_grad():
                    generated_tokens = self.model.generate(
                        **inputs,
                        forced_bos_token_id=target_lang_token,
                        max_length=settings.max_length,
                        num_beams=4,
                        length_penalty=1.0,
                        early_stopping=True,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                
                # Decode results
                translated_texts = self.tokenizer.batch_decode(
                    generated_tokens, 
                    skip_special_tokens=True
                )
                
                total_time = time.time() - start_time
                avg_time = total_time / len(texts)
                
                # Create results
                results = []
                for i, translated_text in enumerate(translated_texts):
                    result = TranslationResult(
                        translated_text=translated_text,
                        source_lang=source_lang,
                        target_lang=target_lang,
                        inference_time=avg_time
                    )
                    results.append(result)
                
                # Update stats
                self.stats["total_requests"] += len(texts)
                self.stats["total_inference_time"] += total_time
                self.stats["average_inference_time"] = (
                    self.stats["total_inference_time"] / self.stats["total_requests"]
                )
                
                logger.debug(f"Batch translation completed in {total_time:.3f}s")
                return results
                
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Batch translation error: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check if the model is healthy and ready."""
        try:
            if not self.model_loaded:
                return False
            
            # Simple test translation
            test_result = await self.translate(
                "Hello", "eng_Latn", "spa_Latn"
            )
            return len(test_result.translated_text) > 0
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_name": settings.model_name,
            "device": self.device,
            "quantization_enabled": settings.quantization_enabled,
            "quantization_type": settings.quantization_type if settings.quantization_enabled else None,
            "model_loaded": self.model_loaded,
            "supported_languages": len(self.tokenizer.lang_code_to_id) if self.tokenizer else 0,
            "max_length": settings.max_length,
            "cache_enabled": settings.enable_cache,
            "stats": self.stats
        }
    
    def clear_cache(self):
        """Clear translation cache."""
        if self._cache:
            self._cache.clear()
            logger.info("Translation cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get translation statistics."""
        return self.stats.copy()
    
    def clear_stats(self):
        """Clear statistics."""
        self.stats = {
            "total_requests": 0,
            "total_inference_time": 0.0,
            "average_inference_time": 0.0,
            "cache_hits": 0,
            "errors": 0,
            "model_load_time": self.stats.get("model_load_time", 0.0)
        }


# Global translation client instance
translation_client = NLLBTranslationClient()
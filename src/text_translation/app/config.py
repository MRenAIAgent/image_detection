"""
Configuration settings for Text Translation API
"""

import os
from typing import List, Optional
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # App basic settings
    app_name: str = "Text Translation API"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 1
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # CORS settings
    cors_origins: List[str] = ["*"]
    cors_methods: List[str] = ["*"]
    cors_headers: List[str] = ["*"]
    
    # Model settings
    model_name: str = "facebook/nllb-200-distilled-600M"
    model_cache_dir: str = "./models"
    device: str = "auto"  # auto, cpu, cuda
    
    # Quantization options
    quantization_enabled: bool = True
    quantization_type: str = "fp16"  # fp16, int8, int4
    
    # Translation settings
    max_length: int = 512
    max_batch_size: int = 8
    batch_timeout: float = 0.1  # seconds
    
    # Request limits
    max_text_length: int = 5000  # characters
    max_concurrent_requests: int = 100
    
    # Cache settings
    enable_cache: bool = True
    cache_ttl: int = 3600  # seconds
    
    # Performance settings
    torch_compile: bool = False  # Enable torch.compile for optimization
    use_bettertransformer: bool = True  # Use optimized transformer implementation
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Language code mappings for NLLB-200
NLLB_LANGUAGE_CODES = {
    # Major languages
    "english": "eng_Latn",
    "spanish": "spa_Latn", 
    "french": "fra_Latn",
    "german": "deu_Latn",
    "italian": "ita_Latn",
    "portuguese": "por_Latn",
    "russian": "rus_Cyrl",
    "chinese": "zho_Hans",  # Simplified Chinese
    "japanese": "jpn_Jpan",
    "korean": "kor_Hang",
    "arabic": "arb_Arab",
    "hindi": "hin_Deva",
    "turkish": "tur_Latn",
    "dutch": "nld_Latn",
    "polish": "pol_Latn",
    "czech": "ces_Latn",
    "hungarian": "hun_Latn",
    "romanian": "ron_Latn",
    "bulgarian": "bul_Cyrl",
    "greek": "ell_Grek",
    "hebrew": "heb_Hebr",
    "thai": "tha_Thai",
    "vietnamese": "vie_Latn",
    "indonesian": "ind_Latn",
    "malay": "zsm_Latn",
    "tagalog": "tgl_Latn",
    "swahili": "swh_Latn",
    "ukrainian": "ukr_Cyrl",
    "danish": "dan_Latn",
    "swedish": "swe_Latn",
    "norwegian": "nob_Latn",
    "finnish": "fin_Latn",
    
    # ISO codes (common alternatives)
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn", 
    "de": "deu_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "ru": "rus_Cyrl",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "ar": "arb_Arab",
    "hi": "hin_Deva",
    "tr": "tur_Latn",
    "nl": "nld_Latn",
    "pl": "pol_Latn",
    "cs": "ces_Latn",
    "hu": "hun_Latn",
    "ro": "ron_Latn",
    "bg": "bul_Cyrl",
    "el": "ell_Grek",
    "he": "heb_Hebr",
    "th": "tha_Thai",
    "vi": "vie_Latn",
    "id": "ind_Latn",
    "ms": "zsm_Latn",
    "tl": "tgl_Latn",
    "sw": "swh_Latn",
    "uk": "ukr_Cyrl",
    "da": "dan_Latn",
    "sv": "swe_Latn",
    "no": "nob_Latn",
    "fi": "fin_Latn",
}

def get_nllb_language_code(language: str) -> Optional[str]:
    """Get NLLB language code from common language name or ISO code."""
    return NLLB_LANGUAGE_CODES.get(language.lower())

def validate_language_pair(source_lang: str, target_lang: str) -> tuple[str, str]:
    """Validate and convert language pair to NLLB codes."""
    source_code = get_nllb_language_code(source_lang)
    target_code = get_nllb_language_code(target_lang)
    
    if not source_code:
        raise ValueError(f"Unsupported source language: {source_lang}")
    if not target_code:
        raise ValueError(f"Unsupported target language: {target_lang}")
    
    return source_code, target_code

def validate_text_length(text: str, max_length: int = 5000) -> bool:
    """Validate text length."""
    return len(text) <= max_length

# Global settings instance
settings = Settings()
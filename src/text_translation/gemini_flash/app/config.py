"""
Configuration settings for Gemini Flash Translation API
"""
import os
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # Application metadata
    app_name: str = "Gemini Flash Translation API"
    app_version: str = "1.0.0"
    description: str = "Fast translation API using Google Gemini Flash"
    debug: bool = Field(default=False, env="DEBUG")
    
    # Server settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8080, env="PORT")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # CORS settings
    cors_origins: List[str] = ["*"]
    cors_methods: List[str] = ["*"]
    cors_headers: List[str] = ["*"]
    
    # Gemini settings
    google_api_key: Optional[str] = Field(default=None, env="GOOGLE_API_KEY")
    gemini_model: str = Field(default="gemini-1.5-flash", env="GEMINI_MODEL")
    
    # Translation settings
    max_text_length: int = Field(default=5000, env="MAX_TEXT_LENGTH")
    max_batch_size: int = Field(default=50, env="MAX_BATCH_SIZE")
    default_temperature: float = Field(default=0.1, env="DEFAULT_TEMPERATURE")
    
    # Rate limiting and performance
    request_timeout: int = Field(default=30, env="REQUEST_TIMEOUT")
    max_concurrent_requests: int = Field(default=100, env="MAX_CONCURRENT_REQUESTS")
    
    # Supported languages (major languages supported by Gemini)
    supported_languages: List[str] = [
        "english", "spanish", "french", "german", "italian", "portuguese", "russian",
        "chinese", "japanese", "korean", "arabic", "hindi", "dutch", "swedish",
        "norwegian", "danish", "finnish", "polish", "czech", "hungarian", "romanian",
        "bulgarian", "croatian", "serbian", "slovenian", "slovak", "latvian", "lithuanian",
        "estonian", "greek", "turkish", "hebrew", "thai", "vietnamese", "indonesian",
        "malay", "filipino", "swahili", "yoruba", "igbo", "hausa", "amharic", "somali",
        "afrikaans", "zulu", "xhosa", "shona", "sesotho", "setswana", "ukrainian",
        "belarusian", "georgian", "armenian", "azerbaijani", "kazakh", "uzbek", "tajik",
        "kyrgyz", "mongolian", "nepali", "bengali", "gujarati", "punjabi", "tamil",
        "telugu", "kannada", "malayalam", "marathi", "urdu", "persian", "pashto",
        "kurdish", "albanian", "macedonian", "bosnian", "montenegrin", "icelandic",
        "irish", "welsh", "scots_gaelic", "basque", "catalan", "galician", "maltese",
        "luxembourgish", "faroese", "sami", "yiddish", "ladino", "esperanto"
    ]
    
    # Language code mappings for better compatibility
    language_mappings: dict = {
        "en": "english", "es": "spanish", "fr": "french", "de": "german",
        "it": "italian", "pt": "portuguese", "ru": "russian", "zh": "chinese",
        "ja": "japanese", "ko": "korean", "ar": "arabic", "hi": "hindi",
        "nl": "dutch", "sv": "swedish", "no": "norwegian", "da": "danish",
        "fi": "finnish", "pl": "polish", "cs": "czech", "hu": "hungarian",
        "ro": "romanian", "bg": "bulgarian", "hr": "croatian", "sr": "serbian",
        "sl": "slovenian", "sk": "slovak", "lv": "latvian", "lt": "lithuanian",
        "et": "estonian", "el": "greek", "tr": "turkish", "he": "hebrew",
        "th": "thai", "vi": "vietnamese", "id": "indonesian", "ms": "malay",
        "tl": "filipino", "sw": "swahili", "yo": "yoruba", "ig": "igbo",
        "ha": "hausa", "am": "amharic", "so": "somali", "af": "afrikaans",
        "zu": "zulu", "xh": "xhosa", "sn": "shona", "st": "sesotho",
        "tn": "setswana", "uk": "ukrainian", "be": "belarusian", "ka": "georgian",
        "hy": "armenian", "az": "azerbaijani", "kk": "kazakh", "uz": "uzbek",
        "tg": "tajik", "ky": "kyrgyz", "mn": "mongolian", "ne": "nepali",
        "bn": "bengali", "gu": "gujarati", "pa": "punjabi", "ta": "tamil",
        "te": "telugu", "kn": "kannada", "ml": "malayalam", "mr": "marathi",
        "ur": "urdu", "fa": "persian", "ps": "pashto", "ku": "kurdish",
        "sq": "albanian", "mk": "macedonian", "bs": "bosnian", "me": "montenegrin",
        "is": "icelandic", "ga": "irish", "cy": "welsh", "gd": "scots_gaelic",
        "eu": "basque", "ca": "catalan", "gl": "galician", "mt": "maltese",
        "lb": "luxembourgish", "fo": "faroese", "se": "sami", "yi": "yiddish",
        "lad": "ladino", "eo": "esperanto"
    }
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
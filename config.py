"""
Configuration management for ConnectIQ RAG API
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    # API Configuration
    app_title: str = "ConnectIQ RAG API"
    app_version: str = "1.0.0"
    
    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    
    # Models
    embedding_model: str = "text-embedding-3-large"
    llm_model: str = "gpt-4o-mini"
    
    # Files
    embeddings_file: str = "embeddings.json"
    
    # RAG Parameters
    default_top_k: int = 5
    min_similarity: float = 0.3
    max_context_chars: int = 4000
    
    # Rate Limiting
    rate_limit: str = "30/minute"
    
    # API Keys (hardcoded per MVP)
    valid_api_keys: dict = {
        "demo_key_12345": {"name": "Demo Client", "tier": "free"},
        "beta_key_67890": {"name": "Beta Tester", "tier": "beta"},
    }
    
    # Validation
    max_question_length: int = 500
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # ← QUESTA RIGA RISOLVE L'ERRORE


settings = Settings()


# Validation at startup
if not settings.openai_api_key:
    raise ValueError(
        "OPENAI_API_KEY not found in environment.\n"
        "Add it to .env file:\n"
        "OPENAI_API_KEY=sk-..."
    )
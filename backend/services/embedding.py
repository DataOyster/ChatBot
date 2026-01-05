"""
Embedding Service - Genera embeddings con OpenAI
"""
import time
from typing import List
from functools import lru_cache
from openai import OpenAI, OpenAIError
from config import settings


class EmbeddingError(Exception):
    """Custom exception per errori embedding"""
    pass


class EmbeddingService:
    """
    Servizio per generare embeddings con OpenAI.
    Include retry logic e caching in-memory.
    """
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.embedding_model
        self.max_retries = 3
    
    def embed_text(self, text: str) -> List[float]:
        """
        Genera embedding per un singolo testo.
        Usa cache in-memory per query ripetute.
        """
        if not text or len(text.strip()) == 0:
            raise EmbeddingError("Cannot embed empty text")
        
        # Tronca se troppo lungo (max 8192 tokens per text-embedding-3-large)
        if len(text) > 30000:  # ~8k tokens
            text = text[:30000]
        
        return self._embed_with_retry(text)
    
    @lru_cache(maxsize=500)  # Cache in-memory per 500 query uniche
    def _embed_with_retry(self, text: str) -> List[float]:
        """Embedding con retry logic"""
        retries = 0
        last_error = None
        
        while retries <= self.max_retries:
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=text,
                )
                return response.data[0].embedding
            
            except OpenAIError as e:
                retries += 1
                last_error = e
                
                # Rate limit o server error → retry con backoff
                if retries <= self.max_retries:
                    wait_time = 2 ** retries  # Exponential backoff
                    time.sleep(wait_time)
                    continue
                else:
                    break
            
            except Exception as e:
                raise EmbeddingError(f"Unexpected error during embedding: {e}")
        
        # Se arriviamo qui, tutti i retry sono falliti
        raise EmbeddingError(
            f"Embedding failed after {self.max_retries} retries. "
            f"Last error: {last_error}"
        )
    
    def is_healthy(self) -> bool:
        """Health check - prova a fare un embedding di test"""
        try:
            self.embed_text("test")
            return True
        except:
            return False
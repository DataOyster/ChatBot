"""
Retrieval Service - Wrapper attorno a Retriever esistente
"""
from typing import List, Dict, Optional
from retriever_fixed import Retriever
from config import settings


class RetrievalError(Exception):
    """Custom exception per errori retrieval"""
    pass


class RetrievalService:
    """
    Wrapper attorno al Retriever esistente.
    Aggiunge validazione e gestione errori.
    """
    
    def __init__(self):
        try:
            self.retriever = Retriever(
                embeddings_file=settings.embeddings_file,
                top_k=settings.default_top_k,
                min_similarity=settings.min_similarity,
            )
        except Exception as e:
            raise RetrievalError(f"Failed to initialize retriever: {e}")
    
    def query(
        self,
        query_embedding: List[float],
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None,
        filters: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Esegue retrieval semantico.
        
        Args:
            query_embedding: Vettore della query
            top_k: Numero risultati (default da config)
            min_similarity: Soglia similarità minima (default da config)
            filters: Filtri opzionali (es. page_type)
        
        Returns:
            Lista di chunks rilevanti con metadata
        """
        if not query_embedding:
            raise RetrievalError("Query embedding cannot be empty")
        
        # Override temporaneo parametri retriever se specificati
        original_top_k = self.retriever.top_k
        original_min_sim = self.retriever.min_similarity
        
        if top_k:
            self.retriever.top_k = top_k
        if min_similarity:
            self.retriever.min_similarity = min_similarity
        
        try:
            results = self.retriever.query(
                query_embedding=query_embedding,
                filters=filters,
            )
            
            return results
        
        except Exception as e:
            raise RetrievalError(f"Retrieval failed: {e}")
        
        finally:
            # Ripristina valori originali
            self.retriever.top_k = original_top_k
            self.retriever.min_similarity = original_min_sim
    
    def is_healthy(self) -> bool:
        """Health check - verifica che embeddings siano caricati"""
        return self.retriever.stats["embeddings_loaded"] > 0
    
    def get_stats(self) -> Dict:
        """Ritorna statistiche retriever"""
        return self.retriever.info()
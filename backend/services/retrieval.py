"""
Retrieval service - runtime semantic search

OPTIMIZED FOR DEMO:
- Higher top_k (8 instead of 5) for better recall
- Lower min_similarity threshold for borderline matches
- Preserves all metadata for debugging
"""

from typing import List, Dict, Optional
import json
import math
from pathlib import Path

from config import settings


class RetrievalError(Exception):
    pass


class RetrievalService:
    """
    Loads precomputed embeddings and performs similarity search.
    
    Demo-optimized configuration:
    - Favors recall over precision
    - Better for staff/partner queries which have weaker semantic signals
    """

    def __init__(self):
        embeddings_path = Path(settings.embeddings_file)

        if not embeddings_path.exists():
            raise RetrievalError(
                f"Embeddings file not found: {embeddings_path}"
            )

        with open(embeddings_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        # Support both formats
        if "embeddings" in payload:
            self.data = payload["embeddings"]
        elif isinstance(payload, list):
            self.data = payload
        else:
            raise RetrievalError("Invalid embeddings file format")

        if not self.data or len(self.data) == 0:
            raise RetrievalError("Embeddings file is empty")

        # Support both "vector" and "embedding" keys
        first_item = self.data[0]
        if "vector" in first_item:
            self.vector_key = "vector"
        elif "embedding" in first_item:
            self.vector_key = "embedding"
        else:
            raise RetrievalError("No vector/embedding key found in data")

        self.vector_dim = len(first_item[self.vector_key])
        self.embeddings_loaded = len(self.data)

        print(f"✅ Loaded {self.embeddings_loaded} embeddings (dim: {self.vector_dim})")

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 8,  # INCREASED from 5 for better recall
        min_similarity: float = 0.20,  # LOWERED from 0.3 for borderline matches
        filters: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Perform semantic search on embeddings.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return (default: 8, optimized for demos)
            min_similarity: Minimum cosine similarity (default: 0.20, tolerant)
            filters: Optional metadata filters
        
        Returns:
            List of matching chunks with metadata and similarity scores
        """
        if len(query_embedding) != self.vector_dim:
            raise RetrievalError(
                f"Query embedding dimension mismatch: "
                f"expected {self.vector_dim}, got {len(query_embedding)}"
            )

        scored = []

        for item in self.data:
            # Apply filters if provided
            if filters:
                skip = False
                for k, v in filters.items():
                    if item.get(k) != v:
                        skip = True
                        break
                if skip:
                    continue

            # Calculate similarity
            similarity = self._cosine_similarity(
                query_embedding, item[self.vector_key]
            )

            # Only include results above threshold
            if similarity >= min_similarity:
                scored.append({
                    "url": item.get("url", ""),
                    "title": item.get("title", ""),
                    "content": item.get("content", ""),
                    "page_type": item.get("page_type", "general"),  # NEW: preserve page type
                    "chunk_id": item.get("chunk_id", ""),  # NEW: for debugging
                    "similarity": similarity,
                })

        # Sort by similarity (descending)
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        
        return scored[:top_k]

    def is_healthy(self) -> bool:
        """Health check"""
        return self.embeddings_loaded > 0

    def get_stats(self) -> Dict:
        """Get service statistics"""
        return {
            "embeddings_loaded": self.embeddings_loaded,
            "vector_dim": self.vector_dim,
        }
"""
Chat endpoint - RAG completo end-to-end
"""
from fastapi import APIRouter, HTTPException, Header, Request, Depends
from pydantic import BaseModel, Field
from typing import List, Optional
from slowapi import Limiter
from slowapi.util import get_remote_address

from config import settings
from services.embedding import EmbeddingService, EmbeddingError
from services.llm import LLMService, LLMError
from services.retrieval import RetrievalService, RetrievalError

# Router
router = APIRouter()

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Services (inizializzati UNA VOLTA all'avvio)
embedding_service = None
llm_service = None
retrieval_service = None


@router.on_event("startup")
async def startup():
    """Inizializza servizi all'avvio"""
    global embedding_service, llm_service, retrieval_service
    
    print("🚀 Initializing services...")
    
    try:
        embedding_service = EmbeddingService()
        print("✅ Embedding service ready")
        
        llm_service = LLMService()
        print("✅ LLM service ready")
        
        retrieval_service = RetrievalService()
        print(f"✅ Retrieval service ready ({retrieval_service.retriever.stats['embeddings_loaded']} embeddings loaded)")
    
    except Exception as e:
        print(f"❌ Failed to initialize services: {e}")
        raise


# ============================================================
# MODELS
# ============================================================

class ChatRequest(BaseModel):
    question: str = Field(..., max_length=500, description="User question")
    top_k: int = Field(5, ge=1, le=10, description="Number of context chunks to retrieve")
    filters: Optional[dict] = Field(None, description="Optional filters (e.g., page_type)")


class Source(BaseModel):
    url: str
    similarity: float
    title: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]
    confidence: str  # "high" | "medium" | "low"
    retrieved_chunks: int


# ============================================================
# AUTH
# ============================================================

def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> dict:
    """Verifica API key (hardcoded per MVP)"""
    if x_api_key not in settings.valid_api_keys:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    return settings.valid_api_keys[x_api_key]


# ============================================================
# ENDPOINTS
# ============================================================

@router.post("/chat", response_model=ChatResponse)
@limiter.limit(settings.rate_limit)
async def chat(
    request: Request,
    req: ChatRequest,
    client_info: dict = Depends(verify_api_key)
):
    """
    RAG-powered Q&A endpoint.
    
    Requires header: X-API-Key
    
    Flow:
    1. Embed user question
    2. Retrieve relevant chunks
    3. Generate answer with LLM
    """
    
    # Validation
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty")
    
    try:
        # 1. EMBEDDING
        query_vector = embedding_service.embed_text(req.question)
        
        # 2. RETRIEVAL
        context_docs = retrieval_service.query(
            query_embedding=query_vector,
            top_k=req.top_k,
            filters=req.filters,
        )
        
        # Se nessun risultato
        if not context_docs:
            return ChatResponse(
                answer="I couldn't find relevant information to answer your question. Please try rephrasing or ask something else.",
                sources=[],
                confidence="low",
                retrieved_chunks=0
            )
        
        # 3. PREPARE CONTEXT
        context = "\n\n---\n\n".join([
            f"[Source: {doc['title'] or doc['url']}]\n{doc['content']}"
            for doc in context_docs
        ])
        
        # 4. GENERATE ANSWER
        answer = llm_service.generate_answer(
            question=req.question,
            context=context
        )
        
        # 5. DETERMINE CONFIDENCE
        top_similarity = context_docs[0]["similarity"]
        if top_similarity > 0.8:
            confidence = "high"
        elif top_similarity > 0.5:
            confidence = "medium"
        else:
            confidence = "low"
        
        # 6. RESPONSE
        return ChatResponse(
            answer=answer,
            sources=[
                Source(
                    url=doc["url"],
                    similarity=doc["similarity"],
                    title=doc["title"] or "Untitled"
                )
                for doc in context_docs
            ],
            confidence=confidence,
            retrieved_chunks=len(context_docs)
        )
    
    except EmbeddingError as e:
        raise HTTPException(503, f"Embedding service error: {str(e)}")
    
    except RetrievalError as e:
        raise HTTPException(503, f"Retrieval error: {str(e)}")
    
    except LLMError as e:
        raise HTTPException(502, f"AI generation error: {str(e)}")
    
    except Exception as e:
        # Log l'errore (in prod usare logger)
        print(f"❌ Unexpected error: {e}")
        raise HTTPException(500, "Internal server error")


@router.get("/stats")
async def get_stats(client_info: dict = Depends(verify_api_key)):
    """Statistiche retriever (richiede auth)"""
    return retrieval_service.get_stats()
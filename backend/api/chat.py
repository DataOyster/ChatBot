"""
Chat endpoint - RAG end-to-end
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

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

embedding_service: EmbeddingService | None = None
llm_service: LLMService | None = None
retrieval_service: RetrievalService | None = None


# -------------------------------------------------
# Startup
# -------------------------------------------------

@router.on_event("startup")
async def startup():
    global embedding_service, llm_service, retrieval_service

    embedding_service = EmbeddingService()
    llm_service = LLMService()
    retrieval_service = RetrievalService()

    print("✅ Services initialized")
    print(f"   Embeddings loaded: {retrieval_service.embeddings_loaded}")


# -------------------------------------------------
# Models
# -------------------------------------------------

class ChatRequest(BaseModel):
    question: str = Field(..., max_length=500)
    top_k: int = Field(5, ge=1, le=10)
    filters: Optional[dict] = None


class Source(BaseModel):
    url: str
    similarity: float
    title: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]
    confidence: str
    retrieved_chunks: int


# -------------------------------------------------
# Auth (MVP)
# -------------------------------------------------

def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    if x_api_key not in settings.valid_api_keys:
        raise HTTPException(401, "Invalid API key")
    return settings.valid_api_keys[x_api_key]


# -------------------------------------------------
# Endpoint
# -------------------------------------------------

@router.post("/chat", response_model=ChatResponse)
@limiter.limit(settings.rate_limit)
async def chat(
    request: Request,
    req: ChatRequest,
    client=Depends(verify_api_key),
):
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty")

    try:
        # 1. Embed query
        query_embedding = embedding_service.embed_text(req.question)

        # 2. Retrieve
        docs = retrieval_service.query(
            query_embedding=query_embedding,
            top_k=req.top_k,
            filters=req.filters,
        )

        if not docs:
            answer = llm_service.generate_answer(
                question=req.question,
                context=""
            )
            return ChatResponse(
                answer=answer,
                sources=[],
                confidence="low",
                retrieved_chunks=0,
            )

        # 3. Build context
        context = "\n\n".join(
            f"{d['content']}" for d in docs
        )

        # 4. Generate
        answer = llm_service.generate_answer(
            question=req.question,
            context=context,
        )

        top_sim = docs[0]["similarity"]
        confidence = (
            "high" if top_sim > 0.8 else
            "medium" if top_sim > 0.5 else
            "low"
        )

        return ChatResponse(
            answer=answer,
            sources=[
                Source(
                    url=d["url"],
                    similarity=d["similarity"],
                    title=d["title"] or "Untitled",
                )
                for d in docs
            ],
            confidence=confidence,
            retrieved_chunks=len(docs),
        )

    except (EmbeddingError, RetrievalError, LLMError) as e:
        raise HTTPException(503, str(e))

    except Exception as e:
        print("❌ Unexpected error:", e)
        raise HTTPException(500, "Internal server error")
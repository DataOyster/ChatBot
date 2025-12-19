from fastapi import APIRouter
from pydantic import BaseModel
from assistant.agent import generate_answer

router = APIRouter()


class ChatRequest(BaseModel):
    message: str


@router.post("/chat")
def chat(request: ChatRequest):
    reply = generate_answer(request.message)
    return {"reply": reply}

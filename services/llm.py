"""
LLM Service - Genera risposte usando OpenAI
"""
from typing import Optional
from openai import OpenAI, OpenAIError
from config import settings


class LLMError(Exception):
    """Custom exception per errori LLM"""
    pass


class LLMService:
    """
    Servizio per generare risposte con GPT.
    Include prompt engineering ottimizzato per RAG.
    """
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.llm_model
        self.max_tokens = 800  # Risposte concise
        self.temperature = 0.3  # Più deterministico
    
    def generate_answer(
        self,
        question: str,
        context: str,
        max_context_chars: Optional[int] = None
    ) -> str:
        """
        Genera risposta basata su context e question.
        
        Args:
            question: Domanda dell'utente
            context: Contesto recuperato dal retriever (chunks concatenati)
            max_context_chars: Limite caratteri context (default da config)
        """
        # Limita lunghezza context per evitare token overflow
        if max_context_chars is None:
            max_context_chars = settings.max_context_chars
        
        if len(context) > max_context_chars:
            context = context[:max_context_chars] + "\n[...context truncated...]"
        
        # Prompt engineering ottimizzato
        system_prompt = """You are a helpful assistant that answers questions based on provided context.

Rules:
- Answer ONLY based on the context provided
- If the context doesn't contain the answer, say "I don't have enough information to answer that question"
- Be concise and direct
- Cite specific details from the context when possible
- Do not make up information"""

        user_prompt = f"""Context:
{context}

Question: {question}

Answer:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            answer = response.choices[0].message.content.strip()
            
            if not answer:
                raise LLMError("LLM returned empty response")
            
            return answer
        
        except OpenAIError as e:
            raise LLMError(f"OpenAI API error: {e}")
        
        except Exception as e:
            raise LLMError(f"Unexpected error during generation: {e}")
    
    def is_healthy(self) -> bool:
        """Health check - prova una chiamata minimale"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5,
            )
            return bool(response.choices[0].message.content)
        except:
            return False
"""
LLM Service - Generate answers using OpenAI

OPTIMIZED FOR DEMOS:
- Confident, factual responses
- No hedging when info is in context
- Clear fallback when info is missing
"""
from typing import Optional
from openai import OpenAI, OpenAIError
from config import settings


class LLMError(Exception):
    """Custom exception for LLM errors"""
    pass


class LLMService:
    """
    Service for generating answers with GPT.
    
    Demo-optimized prompt engineering:
    - Prioritizes directness over hedging
    - Uses factual tone
    - Clear about missing information
    """
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.llm_model
        self.max_tokens = 500  # Concise but complete answers
        self.temperature = 0.2  # More deterministic for factual queries
    
    def generate_answer(
        self,
        question: str,
        context: str,
        max_context_chars: Optional[int] = None
    ) -> str:
        """
        Generate answer based on context and question.
        
        Args:
            question: User's question
            context: Retrieved context from chunks
            max_context_chars: Character limit for context
        
        Returns:
            Generated answer string
        """
        # Limit context length to avoid token overflow
        if max_context_chars is None:
            max_context_chars = settings.max_context_chars
        
        if len(context) > max_context_chars:
            context = context[:max_context_chars] + "\n[...truncated...]"
        
        # OPTIMIZED SYSTEM PROMPT - Anti-evasive, pro-confidence
        system_prompt = """You are a professional conference assistant AI. Your job is to provide accurate, helpful information about the event.

CRITICAL RULES FOR DEMO SUCCESS:

1. WHEN INFORMATION IS IN CONTEXT:
   - Answer DIRECTLY and CONFIDENTLY
   - Use factual, declarative sentences
   - Example: "Wassa Camara is the Project Manager" NOT "It appears that Wassa might be..."
   - DO NOT use hedging language like "seems", "appears", "might be", "I think"
   
2. WHEN INFORMATION IS MISSING:
   - Say clearly: "I don't have that specific information in my database"
   - Offer to help with related info if possible
   - DO NOT say "check the website" - you ARE the website assistant
   
3. FOR CONVERSATIONAL QUERIES:
   - Respond naturally to greetings (hi, hello, how are you)
   - Be friendly but professional
   - Stay on topic (conference)
   
4. FORMATTING:
   - Keep answers concise (2-4 sentences for factual queries)
   - Use bullet points for lists
   - No unnecessary pleasantries in factual answers

5. NEVER:
   - Invent facts not in the context
   - Redirect users to "the website" (you are the interface)
   - Use uncertain language when facts are clear"""

        user_prompt = f"""Context from conference database:
{context}

User question: {question}

Provide a direct, factual answer based on the context. If the answer is in the context, state it confidently. If not, clearly say the information isn't available."""

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
        """Health check - minimal API call"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5,
            )
            return bool(response.choices[0].message.content)
        except:
            return False
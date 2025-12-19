from assistant.prompt import SYSTEM_PROMPT
from assistant.retriever import retrieve_context


def generate_answer(user_question: str) -> str:
    context = retrieve_context(user_question)

    if context is None:
        return "I’m not sure about this yet. I’ll forward your question to the team."

    return context

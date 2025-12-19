import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


conference_data = load_json(
    os.path.join(BASE_DIR, "data", "conferences", "2026_stockholm.json")
)

faq_data = load_json(
    os.path.join(BASE_DIR, "data", "booking", "booking_faq.json")
)


def retrieve_context(user_question: str) -> str:
    q = user_question.lower()

    # keyword-based matching (robust)
    keywords_map = [
        {
            "keywords": ["cancel", "cancellation", "refund"],
            "answer": "Yes, you can cancel your booking up to 14 days before the event."
        },
        {
            "keywords": ["invoice", "receipt"],
            "answer": "Yes, an invoice will be sent automatically after booking."
        },
        {
            "keywords": ["language", "english"],
            "answer": "Yes, the conference language is English."
        },
        {
            "keywords": ["location", "where"],
            "answer": "The conference takes place in Stockholm, Sweden."
        }
    ]

    for item in keywords_map:
        if any(keyword in q for keyword in item["keywords"]):
            return item["answer"]

    return None

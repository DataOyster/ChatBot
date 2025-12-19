from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.chat import router as chat_router

app = FastAPI(title="Conference Booking Assistant")

# CORS configuration (ALLOW LOCAL HTML FILES)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # OK for prototype / demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)


@app.get("/")
def root():
    return {"status": "Conference Assistant running"}

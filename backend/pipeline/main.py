from fastapi import FastAPI
from events import router as events_router

app = FastAPI(title="ConnectIQ Event AI Manager")

app.include_router(events_router, prefix="/events")

@app.get("/")
def health():
    return {"status": "ok"}

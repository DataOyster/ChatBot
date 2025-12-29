from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime
import json, uuid, subprocess
from pathlib import Path

router = APIRouter()
DATA_FILE = Path("storage/events.json")
DATA_FILE.parent.mkdir(exist_ok=True)

class EventCreate(BaseModel):
    name: str
    website: str
    category: str
    language: str
    country: str
    start_date: str
    end_date: str
    indexing_mode: str
    auto_update: bool
    notes: str | None = None

def load_events():
    if not DATA_FILE.exists():
        return []
    return json.loads(DATA_FILE.read_text())

def save_events(events):
    DATA_FILE.write_text(json.dumps(events, indent=2))

@router.get("/")
def list_events():
    return load_events()

@router.post("/")
def create_event(event: EventCreate):
    events = load_events()

    event_id = event.name.lower().replace(" ", "_")
    record = {
        "id": event_id,
        "name": event.name,
        "website": event.website,
        "status": "indexing",
        "created_at": datetime.utcnow().isoformat(),
        **event.dict()
    }

    events.append(record)
    save_events(events)

    # 👉 QUI lanci crawler + loader REALI
    subprocess.Popen([
        "python",
        "crawler_runner.py",
        event.website,
        event_id
    ])

    return record

@router.get("/{event_id}")
def get_event(event_id: str):
    return next(e for e in load_events() if e["id"] == event_id)

"use client";
import { useState } from "react";

export default function EventForm() {
  const [form, setForm] = useState({});

  async function submit() {
    await fetch("http://localhost:8000/events", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(form),
    });
    window.location.href = "/";
  }

  return (
    <div className="max-w-xl space-y-4">
      <input placeholder="Event name" onChange={e => setForm({...form, name:e.target.value})} />
      <input placeholder="Website" onChange={e => setForm({...form, website:e.target.value})} />
      <button onClick={submit} className="bg-indigo-600 px-4 py-2 rounded">
        Create Event
      </button>
    </div>
  );
}

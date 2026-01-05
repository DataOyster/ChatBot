import { getEvents } from "@/lib/api";
import EventCard from "@/components/EventCard";

export default async function Dashboard() {
  const events = await getEvents();

  return (
    <div>
      <h1 className="text-3xl font-bold mb-6">Event AI Manager</h1>
      <div className="grid grid-cols-3 gap-6">
        {events.map((e) => (
          <EventCard key={e.id} event={e} />
        ))}
      </div>
    </div>
  );
}

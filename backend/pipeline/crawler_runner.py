import sys
from loader import AILoader

website = sys.argv[1]
event_id = sys.argv[2]

# Assumi che il crawler produca questo file:
crawler_output = f"data/{event_id}_crawler.json"

# (qui chiami il tuo crawler vero)

loader = AILoader(
    input_file=crawler_output,
    output_file=f"rag/{event_id}.json",
    event_id=event_id
)

loader.run()

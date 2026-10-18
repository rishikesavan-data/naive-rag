import json
from pathlib import Path

path = Path("data/extracted/chunks.jsonl")

with path.open("r", encoding="utf-8") as file:
    chunks = [json.loads(line) for line in file]

for i in range(min(10, len(chunks))):
    print("\n" + "=" * 70)
    print(f"Chunk {i}: {chunks[i]['heading']}")
    print(f"Characters: {chunks[i]['character_count']}")
    print(chunks[i]["text"][-250:])

    if i + 1 < len(chunks):
        print("\nNEXT CHUNK START:")
        print(chunks[i + 1]["text"][:250])
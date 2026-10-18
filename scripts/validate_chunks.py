
import json
from pathlib import Path

path = Path("data/extracted/chunks.jsonl")

with path.open("r", encoding="utf-8") as file:
    chunks = [json.loads(line) for line in file if line.strip()]

print(f"Total chunks: {len(chunks)}")

for i in range(len(chunks) - 1):
    current = chunks[i]["text"].strip()
    following = chunks[i + 1]["text"].strip()

    # Remove the Markdown heading before checking the body.
    current_lines = current.splitlines()
    next_lines = following.splitlines()

    current_body = "\n".join(
        line for line in current_lines
        if not line.strip().startswith("#")
    ).strip()

    next_body = "\n".join(
        line for line in next_lines
        if not line.strip().startswith("#")
    ).strip()

    if not current_body or not next_body:
        continue

    last_word = current_body.split()[-1]
    first_word = next_body.split()[0]

    # Report suspicious fragments, not ordinary sentence boundaries.
    if (
        last_word.startswith("-")
        or first_word.startswith("-")
        or last_word.endswith("-")
    ):
        print(f"\nPossible boundary issue: chunks {i} -> {i + 1}")
        print("Previous ending:", current_body[-100:])
        print("Next beginning:", next_body[:100])

print("\nBoundary inspection complete.")

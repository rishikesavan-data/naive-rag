
from pathlib import Path
import hashlib
import json
import re

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data" / "extracted" / "docling_output.md"
OUTPUT_PATH = BASE_DIR / "data" / "extracted" / "chunks.jsonl"

SOURCE_NAME = "Artificial Intelligence, Machine Learning, and Deep Learning.pdf"
CHUNK_SIZE = 1200
START_HEADING = "INTRODUCTION TO AI"


def clean_markdown(text):
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def split_blocks(markdown):
    blocks = []
    paragraph = []
    table = []
    code = []
    in_code = False

    def flush_paragraph():
        if paragraph:
            value = "\n".join(paragraph).strip()
            if value:
                blocks.append({"type": "text", "text": value})
            paragraph.clear()

    def flush_table():
        if table:
            value = "\n".join(table).strip()
            if value:
                blocks.append({"type": "table", "text": value})
            table.clear()

    for line in markdown.splitlines():
        stripped = line.strip()

        if stripped.startswith(("```", "~~~")):
            flush_paragraph()
            flush_table()
            if not in_code:
                in_code = True
                code.append(line)
            else:
                code.append(line)
                blocks.append({"type": "code", "text": "\n".join(code).strip()})
                code.clear()
                in_code = False
            continue

        if in_code:
            code.append(line)
            continue

        heading = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", stripped)

        if heading:
            flush_paragraph()
            flush_table()
            blocks.append({
                "type": "heading",
                "level": len(heading.group(1)),
                "text": heading.group(2).strip()
            })
        elif stripped.startswith("|"):
            flush_paragraph()
            table.append(line)
        elif not stripped:
            flush_paragraph()
            flush_table()
        else:
            flush_table()
            paragraph.append(line)

    flush_paragraph()
    flush_table()

    if code:
        blocks.append({"type": "code", "text": "\n".join(code).strip()})

    return blocks


def split_long_text(text, limit):
    """Split long text at sentence boundaries, falling back to words."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    pieces = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) > limit:
            if current:
                pieces.append(current)
                current = ""

            words = sentence.split()
            part = ""

            for word in words:
                candidate = f"{part} {word}".strip()
                if part and len(candidate) > limit:
                    pieces.append(part)
                    part = word
                else:
                    part = candidate

            if part:
                current = part
            continue

        candidate = f"{current} {sentence}".strip()

        if current and len(candidate) > limit:
            pieces.append(current)
            current = sentence
        else:
            current = candidate

    if current:
        pieces.append(current)

    return pieces


def build_chunks(markdown):
    blocks = split_blocks(clean_markdown(markdown))

    # Start from the main content heading, excluding front matter.
    start_index = next(
        (
            i for i, block in enumerate(blocks)
            if block["type"] == "heading"
            and block["text"].strip().upper() == START_HEADING.upper()
        ),
        None
    )

    if start_index is None:
        raise ValueError(f"Could not find heading: {START_HEADING}")

    blocks = blocks[start_index:]
    chunks = []
    heading_stack = []
    current_parts = []
    current_length = 0

    def get_heading():
        return " > ".join(name for _, name in heading_stack)

    def get_prefix():
        return "\n".join(
            f"{'#' * level} {name}" for level, name in heading_stack
        )

    def save_chunk():
        nonlocal current_parts, current_length

        body = "\n\n".join(current_parts).strip()
        if not body:
            current_parts = []
            current_length = 0
            return

        prefix = get_prefix()
        content = f"{prefix}\n\n{body}".strip() if prefix else body
        index = len(chunks)
        chunk_id = hashlib.sha256(
            f"{SOURCE_NAME}|{index}|{content}".encode("utf-8")
        ).hexdigest()[:20]

        chunks.append({
            "id": chunk_id,
            "text": content,
            "heading": get_heading(),
            "source": SOURCE_NAME,
            "chunk_index": index,
            "character_count": len(content)
        })

        current_parts = []
        current_length = 0

    for block in blocks:
        if block["type"] == "heading":
            save_chunk()
            level = block["level"]
            heading_stack = [
                item for item in heading_stack if item[0] < level
            ]
            heading_stack.append((level, block["text"]))
            continue

        text = block["text"].strip()
        if not text:
            continue

        prefix = get_prefix()
        limit = max(200, CHUNK_SIZE - len(prefix) - 5)

        pieces = (
            split_long_text(text, limit)
            if len(text) > limit
            else [text]
        )

        for piece in pieces:
            if current_parts and current_length + len(piece) + 2 > limit:
                save_chunk()

            current_parts.append(piece)
            current_length += len(piece) + 2

            # A split piece must not exceed the target on its own.
            if current_length >= limit:
                save_chunk()

    save_chunk()
    return chunks


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Markdown file not found: {INPUT_PATH}")

    markdown = INPUT_PATH.read_text(encoding="utf-8")
    chunks = build_chunks(markdown)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    ids = [chunk["id"] for chunk in chunks]
    oversized = [
        chunk for chunk in chunks
        if chunk["character_count"] > CHUNK_SIZE
    ]

    print("========== CHUNK VALIDATION ==========")
    print(f"Total chunks: {len(chunks)}")
    print(f"Unique IDs: {len(set(ids))}")
    print(f"Duplicate IDs: {len(ids) != len(set(ids))}")
    print(f"Empty chunks: {sum(not c['text'].strip() for c in chunks)}")
    print(f"Chunks over {CHUNK_SIZE} characters: {len(oversized)}")
    print(f"Longest chunk: {max(c['character_count'] for c in chunks)}")
    print(f"Saved to: {OUTPUT_PATH}")

    for chunk in chunks[:3]:
        print("\n" + "-" * 60)
        print(f"Heading: {chunk['heading']}")
        print(f"Characters: {chunk['character_count']}")
        print(chunk["text"][:400])


if __name__ == "__main__":
    main()

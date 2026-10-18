
from pathlib import Path
import re

BASE_DIR = Path(__file__).resolve().parent.parent
MD_PATH = BASE_DIR / "data" / "extracted" / "docling_output.md"

if not MD_PATH.exists():
    raise FileNotFoundError(f"Missing output: {MD_PATH}")

text = MD_PATH.read_text(encoding="utf-8")
lines = text.splitlines()

headings = [
    line for line in lines
    if re.match(r"^\s{0,3}#{1,6}\s+", line)
]

image_markers = re.findall(
    r"<!--\s*image\s*-->", text, flags=re.IGNORECASE
)

markdown_tables = sum(
    1 for line in lines
    if re.match(r"^\s*\|?[\s:|-]+\|[\s:|-]+\|?\s*$", line)
    and "-" in line
)

nonempty_lines = sum(bool(line.strip()) for line in lines)

print("========== INGESTION AUDIT ==========")
print(f"Characters: {len(text):,}")
print(f"Total lines: {len(lines):,}")
print(f"Non-empty lines: {nonempty_lines:,}")
print(f"Markdown headings: {len(headings):,}")
print(f"Possible table separator rows: {markdown_tables:,}")
print(f"Image placeholders: {len(image_markers):,}")

print("\n========== FIRST 20 HEADINGS ==========")
for heading in headings[:20]:
    print(heading)

print("\n========== IMAGE PLACEHOLDER EXAMPLES ==========")
for match in list(re.finditer(
    r"<!--\s*image\s*-->", text, flags=re.IGNORECASE
))[:5]:
    start = max(0, match.start() - 150)
    end = min(len(text), match.end() + 250)
    print(text[start:end].replace("\n", " "))
    print("---")

print("\nAudit completed. Review the output before chunking.")

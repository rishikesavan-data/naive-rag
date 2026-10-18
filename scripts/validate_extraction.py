
from pathlib import Path
import csv
import re

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "extracted"

MARKDOWN_PATH = OUTPUT_DIR / "docling_output.md"
IMAGE_DIR = OUTPUT_DIR / "images"
TABLE_DIR = OUTPUT_DIR / "tables"


def main():
    markdown = MARKDOWN_PATH.read_text(encoding="utf-8")

    images = list(IMAGE_DIR.glob("*"))
    tables = list(TABLE_DIR.glob("*.csv"))

    print("========== EXTRACTION VALIDATION ==========")
    print(f"Markdown characters: {len(markdown):,}")
    print(f"Markdown image references: {len(re.findall(r'!\[.*?\]\(.*?\)', markdown))}")
    print(f"Image files: {len(images)}")
    print(f"Table CSV files: {len(tables)}")

    print("\n========== IMAGE FILES ==========")
    for image in images[:10]:
        print(f"{image.name}: {image.stat().st_size:,} bytes")

    print("\n========== TABLE VALIDATION ==========")
    for table_path in tables[:10]:
        try:
            with table_path.open(
                encoding="utf-8-sig", newline=""
            ) as file:
                rows = list(csv.reader(file))

            nonempty_rows = sum(
                any(cell.strip() for cell in row)
                for row in rows
            )

            print(
                f"{table_path.name}: "
                f"{len(rows)} rows, "
                f"{nonempty_rows} non-empty rows"
            )
        except Exception as exc:
            print(f"{table_path.name}: ERROR - {exc}")

    print("\nValidation finished.")


if __name__ == "__main__":
    main()

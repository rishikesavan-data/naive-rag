
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import PdfFormatOption
from docling_core.types.doc import ImageRefMode


BASE_DIR = Path(__file__).resolve().parent.parent

PDF_PATH = BASE_DIR / "data" / "documents" / (
    "Artificial Intelligence, Machine Learning, and Deep Learning.pdf"
)

OUTPUT_DIR = BASE_DIR / "data" / "extracted"
IMAGE_DIR = OUTPUT_DIR / "images"
MARKDOWN_PATH = OUTPUT_DIR / "docling_output.md"


def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    options = PdfPipelineOptions()
    options.generate_picture_images = True
    options.generate_page_images = True
    options.images_scale = 2.0

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=options
            )
        }
    )

    print(f"Converting: {PDF_PATH.name}")
    result = converter.convert(str(PDF_PATH))
    document = result.document

    # Export Markdown with references to saved images.
    document.save_as_markdown(
        MARKDOWN_PATH,
        artifacts_dir=IMAGE_DIR,
        image_mode=ImageRefMode.REFERENCED,
    )

    # Export individual figure images.
    figure_count = 0

    for index, picture in enumerate(document.pictures, start=1):
        image = picture.get_image(document)

        if image is None:
            continue

        image.save(IMAGE_DIR / f"figure_{index:04d}.png")
        figure_count += 1

    # Export table data separately for validation.
    tables_dir = OUTPUT_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    for index, table in enumerate(document.tables, start=1):
        try:
            dataframe = table.export_to_dataframe(doc=document)
            dataframe.to_csv(
                tables_dir / f"table_{index:04d}.csv",
                index=False,
            )
        except Exception as exc:
            print(f"Table {index} export warning: {exc}")

    print("\n========== INGESTION RESULTS ==========")
    print(f"Text characters: {len(document.export_to_markdown()):,}")
    print(f"Picture elements: {len(document.pictures)}")
    print(f"Images saved: {figure_count}")
    print(f"Table elements: {len(document.tables)}")
    print(f"Markdown: {MARKDOWN_PATH}")
    print(f"Images: {IMAGE_DIR}")
    print(f"Tables: {tables_dir}")


if __name__ == "__main__":
    main()

import logging
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    PdfPipelineOptions,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption


def main() -> None:
    data_dir = Path("data")
    pdf_paths = data_dir.glob("**/*.pdf")

    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True,
        table_structure_options=TableStructureOptions(do_cell_matching=True),
        ocr_options=EasyOcrOptions(),
    )
    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        },
    )

    converter = DocumentConverter()

    for pdf_path in pdf_paths:
        try:
            # Convert PDF to markdown
            res = converter.convert(pdf_path)
            markdown_content = res.document.export_to_markdown()

            # Write markdown to file
            markdown_path = pdf_path.with_suffix(".md")
            markdown_path.write_text(markdown_content, encoding="utf-8")
            logging.info(f"Converted {pdf_path.name}")
        except Exception as e:
            logging.info(f"Error processing {pdf_path.name}: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

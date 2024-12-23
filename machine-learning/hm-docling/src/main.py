import logging
from pathlib import Path

from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter


def main() -> None:
    data_dir = Path("data")
    pdf_paths = data_dir.glob("**/*.pdf")

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.ocr_options = EasyOcrOptions(force_full_page_ocr=True)

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

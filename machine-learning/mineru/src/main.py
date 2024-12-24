import logging
from pathlib import Path

from magic_pdf.config.enums import SupportedPdfParseMethod
from magic_pdf.data.data_reader_writer import FileBasedDataReader, FileBasedDataWriter
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze


def process_pdf(pdf_file_path: Path, output_dir_path: Path) -> None:
    pdf_flle_stem = pdf_file_path.stem
    output_dir_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Processing PDF: {pdf_file_path}")

    image_writer = FileBasedDataWriter(str(output_dir_path))
    markdown_writer = FileBasedDataWriter(str(output_dir_path))
    image_dir = output_dir_path.name

    # Read PDF file into memory
    reader1 = FileBasedDataReader("")
    pdf_bytes = reader1.read(str(pdf_file_path))
    ds = PymuDocDataset(pdf_bytes)

    # Process PDF based on type
    pdf_type = ds.classify()
    logging.info(f"Processing PDF using {pdf_type} mode")

    if pdf_type == SupportedPdfParseMethod.OCR:
        infer_result = ds.apply(doc_analyze, ocr=True)
        pipe_result = infer_result.pipe_ocr_mode(image_writer)
    else:
        infer_result = ds.apply(doc_analyze, ocr=False)
        pipe_result = infer_result.pipe_txt_mode(image_writer)

    # Generate outputs
    model_output = str(output_dir_path / f"{pdf_flle_stem}_model.pdf")
    layout_output = str(output_dir_path / f"{pdf_flle_stem}_layout.pdf")
    spans_output = str(output_dir_path / f"{pdf_flle_stem}_spans.pdf")
    markdown_output = f"{pdf_flle_stem}.md"
    content_list_output = f"{pdf_flle_stem}_content_list.json"

    infer_result.draw_model(model_output)
    pipe_result.draw_layout(layout_output)
    pipe_result.draw_span(spans_output)
    pipe_result.dump_md(markdown_writer, markdown_output, image_dir)
    pipe_result.dump_content_list(markdown_writer, content_list_output, image_dir)

    logging.info(f"All outputs saved to: {output_dir_path}")
    logging.info("Processing completed successfully")


def main() -> None:
    pdf_file_path = Path("data/file.pdf")
    output_dir_path = Path("output")
    process_pdf(pdf_file_path, output_dir_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

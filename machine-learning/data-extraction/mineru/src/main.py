import logging
from pathlib import Path

from magic_pdf.config.enums import SupportedPdfParseMethod
from magic_pdf.data.data_reader_writer import FileBasedDataReader, FileBasedDataWriter
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze

logger = logging.getLogger(__name__)


def process_pdf(pdf_file_path: Path, output_dir_path: Path) -> None:
    pdf_file_stem = pdf_file_path.stem
    output_dir_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Processing PDF: {pdf_file_path}")

    image_writer = FileBasedDataWriter(str(output_dir_path))
    markdown_writer = FileBasedDataWriter(str(output_dir_path))
    image_dir_name = output_dir_path.name

    # Read PDF file into memory
    file_reader = FileBasedDataReader("")
    pdf_bytes = file_reader.read(str(pdf_file_path))
    dataset = PymuDocDataset(pdf_bytes)

    # Process PDF based on type
    pdf_parse_method = dataset.classify()
    logger.info(f"Processing PDF using {pdf_parse_method} mode")

    if pdf_parse_method == SupportedPdfParseMethod.OCR:
        infer_result = dataset.apply(doc_analyze, ocr=True)
        pipe_result = infer_result.pipe_ocr_mode(image_writer)
    else:
        infer_result = dataset.apply(doc_analyze, ocr=False)
        pipe_result = infer_result.pipe_txt_mode(image_writer)

    # Generate outputs
    model_output_path = output_dir_path / Path(f"{pdf_file_stem}_model.pdf")
    layout_output_path = output_dir_path / Path(f"{pdf_file_stem}_layout.pdf")
    spans_output_path = output_dir_path / Path(f"{pdf_file_stem}_spans.pdf")
    markdown_output_name = f"{pdf_file_stem}.md"
    content_list_output_name = f"{pdf_file_stem}_content_list.json"

    infer_result.draw_model(str(model_output_path))
    pipe_result.draw_layout(str(layout_output_path))
    pipe_result.draw_span(str(spans_output_path))
    pipe_result.dump_md(markdown_writer, markdown_output_name, image_dir_name)
    pipe_result.dump_content_list(
        markdown_writer,
        content_list_output_name,
        image_dir_name,
    )

    logger.info("Processing completed successfully")


def main() -> None:
    data_dir = Path("data")
    pdf_paths = data_dir.glob("**/*.pdf")
    base_output_dir = Path("output")

    for pdf_path in pdf_paths:
        output_dir_path = base_output_dir / pdf_path.stem
        process_pdf(pdf_path, output_dir_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()

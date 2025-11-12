import logging

from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)


def download_model() -> None:
    logger.info("Downloading model...")
    model_path = snapshot_download("Qwen/Qwen2.5-0.5B-Instruct")
    logger.info(f"Model downloaded successfully to: {model_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    download_model()

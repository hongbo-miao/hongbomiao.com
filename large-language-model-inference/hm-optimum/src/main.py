import logging
import time

from optimum.pipelines import pipeline

logger = logging.getLogger(__name__)

ITERATION_COUNT = 10000


def main() -> None:
    # accelerator="ort" handles the ONNX conversion and runtime mapping
    classifier = pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        accelerator="ort",
    )

    text = "I've been waiting for this course my whole life."

    start_time = time.time()
    for _ in range(ITERATION_COUNT):
        classifier(text)
    total_time = time.time() - start_time

    logger.info(f"Total time for {ITERATION_COUNT} iterations: {total_time} seconds")
    logger.info(f"Average time per iteration: {total_time / ITERATION_COUNT} seconds")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()

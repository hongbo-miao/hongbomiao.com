import logging
import time

from transformers import pipeline

logger = logging.getLogger(__name__)

ITERATION_COUNT = 10000


def main() -> None:
    classifier = pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
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

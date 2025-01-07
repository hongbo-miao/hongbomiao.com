import logging

from transformers import pipeline

logger = logging.getLogger(__name__)


def main() -> None:
    classifier = pipeline("sentiment-analysis")
    logger.info(classifier("I've been waiting for this course my whole life."))
    logger.info(classifier("I've been waiting for so long for this package to ship."))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()

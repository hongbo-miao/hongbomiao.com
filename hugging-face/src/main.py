import logging

from transformers import pipeline


def main() -> None:
    classifier = pipeline("sentiment-analysis")
    logging.info(classifier("I've been waiting for this course my whole life."))
    logging.info(classifier("I've been waiting for so long for this package to ship."))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

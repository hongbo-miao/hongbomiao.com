import logging

import jiwer

logger = logging.getLogger(__name__)


def main() -> None:
    reference = "hello world"
    hypothesis = "hello duck"

    word_error_rate = jiwer.wer(reference, hypothesis)
    match_error_rate = jiwer.mer(reference, hypothesis)
    word_information_lost = jiwer.wil(reference, hypothesis)
    word_information_preserved = jiwer.wip(reference, hypothesis)
    character_error_rate = jiwer.cer(reference, hypothesis)

    logger.info(f"Reference: {reference}")
    logger.info(f"Hypothesis: {hypothesis}")
    logger.info(f"Word Error Rate (WER): {word_error_rate:.2%}")
    logger.info(f"Match Error Rate (MER): {match_error_rate:.2%}")
    logger.info(f"Word Information Lost (WIL): {word_information_lost:.2%}")
    logger.info(f"Word Information Preserved (WIP): {word_information_preserved:.2%}")
    logger.info(f"Character Error Rate (CER): {character_error_rate:.2%}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()

import logging

from utils.get_refund_policy import get_refund_policy

logger = logging.getLogger(__name__)


def main() -> None:
    question = "What if these shoes don't fit?"
    logger.info(f"Question: {question}")
    logger.info(f"Answer: {get_refund_policy(question, refund_day_number=30)}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()

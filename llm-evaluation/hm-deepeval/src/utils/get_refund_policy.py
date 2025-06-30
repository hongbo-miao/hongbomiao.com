import logging

logger = logging.getLogger(__name__)


def get_refund_policy(question: str, refund_day_number: int) -> str:  # noqa: ARG001
    return f"You have {refund_day_number} days to get a full refund at no extra cost."

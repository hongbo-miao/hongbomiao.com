import logging

import pulsar

logger = logging.getLogger(__name__)


def handle_send_result(
    result: pulsar.Result,
    message_id: pulsar.MessageId,
) -> None:
    if result != pulsar.Result.Ok:
        logger.error(f"Failed to produce validated message {message_id}: {result}")

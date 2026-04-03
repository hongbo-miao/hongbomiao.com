import logging

import pulsar
from shared.telemetry.utils.process_telemetry_message import process_telemetry_message

logger = logging.getLogger(__name__)


def consume_telemetry_messages(
    consumer: pulsar.Consumer,
    subscriber_id: str,
    receive_timeout_ms: int,
) -> None:
    while True:
        try:
            message = consumer.receive(timeout_millis=receive_timeout_ms)
        except pulsar.Timeout:
            continue
        except pulsar.PulsarException:
            logger.exception("Failed to receive telemetry message")
            continue
        is_processed = process_telemetry_message(message, subscriber_id)
        if is_processed:
            consumer.acknowledge(message)
        else:
            consumer.negative_acknowledge(message)

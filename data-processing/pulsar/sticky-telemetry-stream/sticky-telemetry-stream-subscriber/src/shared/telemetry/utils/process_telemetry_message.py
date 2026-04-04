import logging

import pulsar

logger = logging.getLogger(__name__)


def process_telemetry_message(
    message: pulsar.Message,
    subscriber_id: str,
) -> bool:
    try:
        partition_key = message.partition_key()
        publisher_id = partition_key or "unknown"

        telemetry = message.value()
        timestamp = telemetry.timestamp

        entries: dict[str, float | None] = {}
        for entry in telemetry.entries:
            entries[entry.name] = entry.value

        telemetry_log = {
            "publisher_id": publisher_id,
            "timestamp": timestamp,
            "entries": entries,
            "topic": message.topic_name(),
        }
        logger.info(
            f"Subscriber {subscriber_id}: {telemetry_log}",
        )
    except Exception:
        logger.exception(
            f"Subscriber {subscriber_id}: error processing message {message.message_id()}",
        )
        return False
    else:
        return True

import logging

import pulsar
from shared.telemetry.schemas.telemetry_reading import TelemetryReading

logger = logging.getLogger(__name__)


def publish_reading(producer: pulsar.Producer, reading: TelemetryReading) -> None:
    producer.send_async(
        reading.model_dump_json().encode(),
        partition_key=reading.device_id,
        callback=lambda result, message_id: handle_publish_result(
            result,
            message_id,
            reading.device_id,
        ),
    )


def handle_publish_result(
    result: pulsar.Result,
    message_id: pulsar.MessageId | None,
    device_id: str,
) -> None:
    if result != pulsar.Result.Ok:
        logger.error(f"Failed to publish reading for device {device_id}: {result}")
        return
    logger.info(f"Published reading for device {device_id}: message_id={message_id}")

import logging
import time
from datetime import UTC, datetime
from functools import partial

import pulsar
from sticky_telemetry_stream_schema.telemetry_record import EntryRecord, TelemetryRecord

logger = logging.getLogger(__name__)

SENSOR_NAMES: list[str] = [
    "temperature_c",
    "humidity_pct",
]


def handle_send_result(
    publisher_id: str,
    result: pulsar.Result,
    _message_id: pulsar.MessageId,
) -> None:
    if result != pulsar.Result.Ok:
        logger.error(
            f"Publisher {publisher_id}: failed to send message: {result}",
        )


def publish_telemetry_stream(
    producer: pulsar.Producer,
    publisher_id: str,
    publish_interval_s: float,
) -> int:
    published_sample_count = 0
    try:
        sample_index = 0
        while True:
            timestamp = datetime.now(UTC).isoformat()
            sensor_entries: list[tuple[str, float | None]] = [
                (sensor_name, float(sample_index)) for sensor_name in SENSOR_NAMES
            ]

            telemetry = TelemetryRecord(
                timestamp=timestamp,
                entries=[
                    EntryRecord(name=entry_name, value=entry_value)
                    for entry_name, entry_value in sensor_entries
                ],
            )

            published_sample_count += 1

            producer.send_async(
                telemetry,
                partition_key=publisher_id,
                callback=partial(handle_send_result, publisher_id),
            )

            if published_sample_count % 10 == 0:
                logger.info(
                    f"Publisher {publisher_id}: published {published_sample_count} telemetry samples",
                )

            time.sleep(publish_interval_s)
            sample_index += 1

    except KeyboardInterrupt:
        logger.info(f"Publisher {publisher_id}: telemetry streaming cancelled")
    except Exception:
        logger.exception("Failed to publish telemetry stream")
        raise

    return published_sample_count

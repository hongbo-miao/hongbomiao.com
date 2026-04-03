import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import capnp
import pulsar

if TYPE_CHECKING:
    from capnp_types.telemetry import (
        EntryBuilder,
        TelemetryBuilder,
    )

logger = logging.getLogger(__name__)

SENSOR_NAMES: list[str] = [
    "temperature_c",
    "humidity_pct",
    "wind_speed_mps",
    "wind_direction_deg",
    "pressure_hpa",
]

TELEMETRY_SCHEMA = capnp.load(
    str(Path(__file__).parents[5] / "schemas" / "telemetry.capnp"),
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

            telemetry: TelemetryBuilder = TELEMETRY_SCHEMA.Telemetry.new_message()
            telemetry.timestamp = timestamp

            sensor_entries_builder = telemetry.init(
                "entries",
                len(sensor_entries),
            )

            for index, (entry_name, entry_value) in enumerate(sensor_entries):
                entry: EntryBuilder = sensor_entries_builder[index]
                entry.name = entry_name
                if entry_value is None:
                    entry.data.missing = None
                else:
                    entry.data.value = entry_value

            telemetry_payload_bytes = telemetry.to_bytes()

            published_sample_count += 1
            message_id = producer.send(
                content=telemetry_payload_bytes,
                partition_key=publisher_id,
            )

            if published_sample_count % 10 == 0:
                logger.info(
                    f"Publisher {publisher_id}: published {published_sample_count} telemetry samples "
                    f"(latest message id: {message_id})",
                )

            time.sleep(publish_interval_s)
            sample_index += 1

    except KeyboardInterrupt:
        logger.info(f"Publisher {publisher_id}: telemetry streaming cancelled")
    except Exception:
        logger.exception("Failed to publish telemetry stream")
        raise

    return published_sample_count

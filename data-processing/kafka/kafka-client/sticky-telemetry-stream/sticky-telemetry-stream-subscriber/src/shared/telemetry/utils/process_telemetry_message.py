import logging
from pathlib import Path
from typing import TYPE_CHECKING

import capnp
from confluent_kafka import Message

if TYPE_CHECKING:
    from capnp_types.telemetry import TelemetryReader

logger = logging.getLogger(__name__)

TELEMETRY_SCHEMA = capnp.load(
    str(Path(__file__).parents[5] / "schemas" / "telemetry.capnp"),
)


def process_telemetry_message(
    message: Message,
    subscriber_id: str,
) -> None:
    try:
        key_bytes = message.key()
        publisher_id = key_bytes.decode() if key_bytes else "unknown"
        telemetry: TelemetryReader
        with TELEMETRY_SCHEMA.Telemetry.from_bytes(message.value()) as telemetry:
            telemetry_log = {
                "publisher_id": publisher_id,
                "partition": message.partition(),
                "timestamp_ns": telemetry.timestampNs,
                "temperature_c": telemetry.temperatureC,
                "humidity_pct": telemetry.humidityPct,
            }
            logger.info(
                f"Subscriber {subscriber_id}: {telemetry_log}",
            )

    except Exception:
        logger.exception(
            f"Subscriber {subscriber_id}: error processing message from partition {message.partition()}",
        )

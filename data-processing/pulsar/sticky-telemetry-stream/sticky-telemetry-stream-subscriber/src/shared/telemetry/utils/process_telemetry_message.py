import logging
from pathlib import Path
from typing import TYPE_CHECKING

import capnp
import pulsar

if TYPE_CHECKING:
    from capnp_types.telemetry import (
        EntryReader,
        TelemetryReader,
    )

logger = logging.getLogger(__name__)

TELEMETRY_SCHEMA = capnp.load(
    str(Path(__file__).parents[5] / "schemas" / "telemetry.capnp"),
)


def process_telemetry_message(
    message: pulsar.Message,
    subscriber_id: str,
) -> bool:
    try:
        partition_key = message.partition_key()
        publisher_id = partition_key or "unknown"
        telemetry: TelemetryReader
        with TELEMETRY_SCHEMA.Telemetry.from_bytes(message.data()) as telemetry:
            timestamp = telemetry.timestamp

            entries: dict[str, float | None] = {}
            entry: EntryReader
            for entry in telemetry.entries:
                data_type = entry.data.which()
                if data_type == "value":
                    entries[entry.name] = entry.data.value
                else:
                    entries[entry.name] = None

            telemetry_log = {
                "publisher_id": publisher_id,
                "topic": message.topic_name(),
                "timestamp": timestamp,
                "entries": entries,
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

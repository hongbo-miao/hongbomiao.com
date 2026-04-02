import logging
from pathlib import Path
from typing import TYPE_CHECKING

import capnp
import nats.aio.msg

if TYPE_CHECKING:
    from capnp_types.telemetry import (
        EntryReader,
        TelemetryReader,
    )

logger = logging.getLogger(__name__)

TELEMETRY_SCHEMA = capnp.load(
    str(Path(__file__).parents[5] / "schemas" / "telemetry.capnp"),
)


def extract_publisher_id(subject: str) -> str:
    return subject.split(".")[1]


async def process_telemetry_message(
    message: nats.aio.msg.Msg,
    subscriber_id: str,
) -> None:
    try:
        publisher_id = extract_publisher_id(message.subject)
        telemetry: TelemetryReader
        with TELEMETRY_SCHEMA.Telemetry.from_bytes(message.data) as telemetry:
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
                "timestamp": timestamp,
                "entries": entries,
            }
            logger.info(
                f"Subscriber {subscriber_id}: {telemetry_log}",
            )

        await message.ack()
    except Exception:
        logger.exception(
            f"Subscriber {subscriber_id}: error processing message from '{message.subject}'",
        )
        await message.nak()

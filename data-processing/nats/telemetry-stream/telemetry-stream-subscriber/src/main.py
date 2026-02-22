import asyncio
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING

import capnp
import nats
from nats.aio.msg import Msg

if TYPE_CHECKING:
    from capnp_types.telemetry import (
        EntryReader,
        TelemetryReader,
    )

logger = logging.getLogger(__name__)

NATS_URL = "nats://localhost:4222"
SUBJECT_PREFIX = "SENSOR_TELEMETRY_STREAMS"
STREAM_SUBJECT = f"{SUBJECT_PREFIX}.random"
QUEUE_GROUP = "telemetry_queue_group"

TELEMETRY_SCHEMA = capnp.load(
    str(Path(__file__).parents[2] / "schemas" / "telemetry.capnp"),
)


def create_telemetry_message_handler() -> Callable[[Msg], Awaitable[None]]:
    async def handle_telemetry_message(message: Msg) -> None:
        try:
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
                    "timestamp": timestamp,
                    "Entries": entries,
                }
                logger.info(f"Telemetry sample: {telemetry_log}")

            await message.ack()
        except Exception:
            logger.exception(f"Error processing message from '{message.subject}'")
            await message.nak()

    return handle_telemetry_message


async def main() -> None:
    nats_client = None
    try:
        logger.info(f"Connecting to NATS at {NATS_URL}")
        nats_client = await nats.connect(NATS_URL)
        jetstream_context = nats_client.jetstream()

        message_handler = create_telemetry_message_handler()

        await jetstream_context.subscribe(
            STREAM_SUBJECT,
            queue=QUEUE_GROUP,
            cb=message_handler,
            manual_ack=True,
        )
        logger.info(
            f"Subscribed to '{STREAM_SUBJECT}' in queue group '{QUEUE_GROUP}'",
        )

        await asyncio.Event().wait()

    except Exception:
        logger.exception("Subscriber error")
        raise
    finally:
        if nats_client:
            await nats_client.drain()
            logger.info("NATS connection closed")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    asyncio.run(main())

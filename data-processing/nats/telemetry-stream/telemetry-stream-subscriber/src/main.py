import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import capnp
import nats

if TYPE_CHECKING:
    from capnp_types.telemetry import (
        EntryReader,
        TelemetryReader,
    )

logger = logging.getLogger(__name__)

NATS_URL = "nats://localhost:4222"
NATS_STREAM_NAME = "SENSOR_TELEMETRY_STREAMS"
SUBJECT_FILTER = f"{NATS_STREAM_NAME}.random"
DURABLE_NAME = "telemetry_queue_group"
FETCH_BATCH_SIZE = 100

TELEMETRY_SCHEMA = capnp.load(
    str(Path(__file__).parents[2] / "schemas" / "telemetry.capnp"),
)


async def process_telemetry_message(message: nats.aio.msg.Msg) -> None:
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


async def main() -> None:
    nats_client = None
    try:
        logger.info(f"Connecting to NATS at {NATS_URL}")
        nats_client = await nats.connect(NATS_URL)
        jetstream_context = nats_client.jetstream()

        consumer = await jetstream_context.pull_subscribe(
            SUBJECT_FILTER,
            durable=DURABLE_NAME,
        )
        logger.info(
            f"Created pull consumer for '{SUBJECT_FILTER}' with durable name '{DURABLE_NAME}'",
        )

        while True:
            try:
                messages = await consumer.fetch(batch=FETCH_BATCH_SIZE, timeout=5)
                for message in messages:
                    await process_telemetry_message(message)
            except nats.errors.TimeoutError:
                continue

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

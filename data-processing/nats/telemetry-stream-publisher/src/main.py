import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from random import SystemRandom
from uuid import uuid4

import capnp
import nats
from nats.js.client import JetStreamContext

logger = logging.getLogger(__name__)

NATS_URL = "nats://localhost:4222"
SUBJECT_PREFIX = "SENSOR_TELEMETRY_STREAMS"
STREAM_SUBJECT = f"{SUBJECT_PREFIX}.random"
DATA_MISSING_PROBABILITY = 0.35
PUBLISH_INTERVAL_SECONDS = 1.0

SENSOR_DEFINITIONS: dict[str, tuple[float, float]] = {
    "temperature_c": (-20.0, 45.0),
    "humidity_pct": (5.0, 95.0),
    "wind_speed_mps": (0.0, 30.0),
    "wind_direction_deg": (0.0, 360.0),
    "pressure_hpa": (950.0, 1050.0),
}

TELEMETRY_SCHEMA = capnp.load(str(Path(__file__).with_name("telemetry.capnp")))


async def publish_random_telemetry_stream(
    jetstream_context: JetStreamContext,
    subject: str,
) -> int:
    published_sample_count = 0
    random_number_generator = SystemRandom()
    try:
        # Generate and stream random telemetry samples
        while True:
            timestamp = datetime.now(UTC).isoformat()
            sensor_entries: list[tuple[str, float | None]] = []

            for entry_name, value_range in SENSOR_DEFINITIONS.items():
                minimum_value, maximum_value = value_range
                if random_number_generator.random() < DATA_MISSING_PROBABILITY:
                    sensor_entries.append((entry_name, None))
                    continue

                value = random_number_generator.uniform(minimum_value, maximum_value)
                sensor_entries.append((entry_name, round(value, 2)))

            telemetry = TELEMETRY_SCHEMA.Telemetry.new_message()
            telemetry.timestamp = timestamp

            sensor_entries_builder = telemetry.init(
                "entries",
                len(sensor_entries),
            )

            for index, (entry_name, entry_value) in enumerate(sensor_entries):
                entry = sensor_entries_builder[index]
                entry.name = entry_name
                if entry_value is None:
                    entry.data.missing = None
                else:
                    entry.data.value = entry_value

            telemetry_payload_bytes = telemetry.to_bytes()

            published_sample_count += 1
            publish_acknowledgement = await jetstream_context.publish(
                subject,
                telemetry_payload_bytes,
                headers={
                    "Nats-Msg-Id": str(uuid4()),
                },
            )

            if published_sample_count % 50 == 0:
                logger.info(
                    f"Published {published_sample_count} telemetry samples "
                    f"(latest sequence: {publish_acknowledgement.seq})",
                )

            await asyncio.sleep(PUBLISH_INTERVAL_SECONDS)

    except asyncio.CancelledError:
        logger.info("Telemetry streaming cancelled")
        raise
    except Exception:
        logger.exception("Failed to publish telemetry stream")
        raise

    return published_sample_count


async def main() -> None:
    nats_client = None
    try:
        logger.info(f"Connecting to NATS at {NATS_URL}")
        nats_client = await nats.connect(NATS_URL)
        jetstream_context = nats_client.jetstream()

        published_sample_count = await publish_random_telemetry_stream(
            jetstream_context=jetstream_context,
            subject=STREAM_SUBJECT,
        )

        logger.info(
            f"Telemetry streaming completed with {published_sample_count} samples published",
        )

    except Exception:
        logger.exception("Publisher error")
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

import asyncio
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

import capnp
from confluent_kafka.aio import AIOProducer

if TYPE_CHECKING:
    from capnp_types.telemetry import TelemetryBuilder

logger = logging.getLogger(__name__)

TELEMETRY_SCHEMA = capnp.load(
    str(Path(__file__).parents[5] / "schemas" / "telemetry.capnp"),
)


async def publish_telemetry_stream(
    producer: AIOProducer,
    topic: str,
    publisher_id: str,
    publish_interval_s: float,
) -> NoReturn:
    published_sample_count = 0
    try:
        sample_index = 0
        while True:
            timestamp_ns = time.time_ns()

            telemetry: TelemetryBuilder = TELEMETRY_SCHEMA.Telemetry.new_message()
            telemetry.timestampNs = timestamp_ns
            telemetry.temperatureC = float(sample_index)
            telemetry.humidityPct = float(sample_index)

            telemetry_payload_bytes = telemetry.to_bytes()

            published_sample_count += 1
            delivered_message_future = await producer.produce(
                topic=topic,
                value=telemetry_payload_bytes,
                key=publisher_id.encode(),
            )
            delivered_message = await delivered_message_future

            if published_sample_count % 10 == 0:
                logger.info(
                    f"Publisher {publisher_id}: published {published_sample_count} telemetry samples "
                    f"(latest partition: {delivered_message.partition()}, offset: {delivered_message.offset()})",
                )

            await asyncio.sleep(publish_interval_s)
            sample_index += 1

    except asyncio.CancelledError:
        logger.info(f"Publisher {publisher_id}: telemetry streaming cancelled")
        raise
    except Exception:
        logger.exception("Failed to publish telemetry stream")
        raise

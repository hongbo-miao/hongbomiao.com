import asyncio
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn
from uuid import uuid4

import capnp
from nats.js.client import JetStreamContext

if TYPE_CHECKING:
    from capnp_types.telemetry import TelemetryBuilder

logger = logging.getLogger(__name__)

TELEMETRY_SCHEMA = capnp.load(
    str(Path(__file__).parents[5] / "schemas" / "telemetry.capnp"),
)


async def publish_telemetry_stream(
    jetstream_context: JetStreamContext,
    subject: str,
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
            publish_acknowledgement = await jetstream_context.publish(
                subject,
                telemetry_payload_bytes,
                headers={
                    "Nats-Msg-Id": str(uuid4()),
                },
            )

            if published_sample_count % 10 == 0:
                logger.info(
                    f"Publisher {publisher_id}: published {published_sample_count} telemetry samples "
                    f"(latest sequence: {publish_acknowledgement.seq})",
                )

            await asyncio.sleep(publish_interval_s)
            sample_index += 1

    except asyncio.CancelledError:
        logger.info(f"Publisher {publisher_id}: telemetry streaming cancelled")
        raise
    except Exception:
        logger.exception("Failed to publish telemetry stream")
        raise

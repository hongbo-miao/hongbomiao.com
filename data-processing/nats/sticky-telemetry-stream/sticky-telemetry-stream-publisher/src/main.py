import asyncio
import logging
import os
from uuid import uuid4

import nats
from shared.telemetry.services.publish_telemetry_stream import (
    publish_telemetry_stream,
)

logger = logging.getLogger(__name__)

NATS_URL = os.environ.get("NATS_URL", "nats://localhost:4222")
SUBJECT_PREFIX = "SENSOR_TELEMETRY_STREAMS"
PUBLISH_INTERVAL_S = 1.0


async def main() -> None:
    nats_client = None
    try:
        publisher_id = str(uuid4())[:8]
        logger.info(f"Publisher {publisher_id}: connecting to NATS at {NATS_URL}")
        nats_client = await nats.connect(NATS_URL)
        jetstream_context = nats_client.jetstream()

        await publish_telemetry_stream(
            jetstream_context=jetstream_context,
            subject=f"{SUBJECT_PREFIX}.{publisher_id}",
            publisher_id=publisher_id,
            publish_interval_s=PUBLISH_INTERVAL_S,
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

import logging
import os
from uuid import uuid4

import pulsar
from shared.telemetry.services.publish_telemetry_stream import (
    publish_telemetry_stream,
)

logger = logging.getLogger(__name__)

PULSAR_SERVICE_URL = os.environ.get("PULSAR_SERVICE_URL", "pulsar://localhost:6650")
PULSAR_TOPIC = "persistent://public/default/sensor-telemetry-streams"
PUBLISH_INTERVAL_S = 1.0


def main() -> None:
    client = None
    try:
        publisher_id = str(uuid4())[:8]
        logger.info(
            f"Publisher {publisher_id}: connecting to Pulsar at {PULSAR_SERVICE_URL}",
        )
        client = pulsar.Client(PULSAR_SERVICE_URL)
        producer = client.create_producer(PULSAR_TOPIC)

        published_sample_count = publish_telemetry_stream(
            producer=producer,
            publisher_id=publisher_id,
            publish_interval_s=PUBLISH_INTERVAL_S,
        )

        logger.info(
            f"Publisher {publisher_id}: telemetry streaming completed with {published_sample_count} samples published",
        )

    except Exception:
        logger.exception("Publisher error")
        raise
    finally:
        if client:
            client.close()
            logger.info("Pulsar client closed")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()

import asyncio
import logging
import os
from uuid import uuid4

from confluent_kafka.aio import AIOProducer
from shared.telemetry.services.publish_telemetry_stream import (
    publish_telemetry_stream,
)

logger = logging.getLogger(__name__)

KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = "sensor-telemetry-streams"
PUBLISH_INTERVAL_S = 1.0


async def main() -> None:
    producer = None
    try:
        publisher_id = str(uuid4())[:8]
        logger.info(
            f"Publisher {publisher_id}: connecting to Kafka at {KAFKA_BOOTSTRAP_SERVERS}",
        )
        producer = AIOProducer({"bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS})

        await publish_telemetry_stream(
            producer=producer,
            topic=KAFKA_TOPIC,
            publisher_id=publisher_id,
            publish_interval_s=PUBLISH_INTERVAL_S,
        )

    except Exception:
        logger.exception("Publisher error")
        raise
    finally:
        if producer:
            await producer.close()
            logger.info("Kafka producer closed")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    asyncio.run(main())

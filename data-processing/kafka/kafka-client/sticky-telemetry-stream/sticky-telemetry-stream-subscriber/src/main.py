import asyncio
import logging
import os
from pathlib import Path
from uuid import uuid4

from confluent_kafka.aio import AIOConsumer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroDeserializer
from shared.telemetry.utils.consume_telemetry_messages import (
    consume_telemetry_messages,
)

logger = logging.getLogger(__name__)

KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
SCHEMA_REGISTRY_URL = os.environ.get("SCHEMA_REGISTRY_URL", "http://localhost:8081")
KAFKA_TOPIC = "sensor-telemetry-streams"
CONSUMER_GROUP_ID = "telemetry-subscriber-group"
POLL_TIMEOUT_S = 1.0

TELEMETRY_AVRO_SCHEMA = (
    Path(__file__).parents[2] / "schemas" / "telemetry.avsc"
).read_text()


async def main() -> None:
    consumer = None
    try:
        subscriber_id = str(uuid4())[:8]
        logger.info(
            f"Subscriber {subscriber_id}: connecting to Kafka at {KAFKA_BOOTSTRAP_SERVERS}",
        )
        consumer = AIOConsumer(
            {
                "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
                "group.id": CONSUMER_GROUP_ID,
                "auto.offset.reset": "earliest",
                "partition.assignment.strategy": "cooperative-sticky",
                "session.timeout.ms": 2000,
                "heartbeat.interval.ms": 500,
                "auto.commit.interval.ms": 5000,
            },
        )
        await consumer.subscribe([KAFKA_TOPIC])

        logger.info(
            f"Subscriber {subscriber_id}: joined consumer group '{CONSUMER_GROUP_ID}' "
            f"with cooperative-sticky partition assignment",
        )

        schema_registry_client = SchemaRegistryClient({"url": SCHEMA_REGISTRY_URL})
        avro_deserializer = AvroDeserializer(
            schema_registry_client,
            TELEMETRY_AVRO_SCHEMA,
        )

        await consume_telemetry_messages(
            consumer,
            subscriber_id,
            POLL_TIMEOUT_S,
            avro_deserializer,
        )

    except Exception:
        logger.exception("Subscriber error")
        raise
    finally:
        if consumer:
            await consumer.close()
            logger.info("Kafka consumer closed")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    asyncio.run(main())

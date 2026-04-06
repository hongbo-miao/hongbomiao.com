import asyncio
import logging
import time
from typing import NoReturn

from confluent_kafka.aio import AIOProducer
from confluent_kafka.schema_registry.avro import AvroSerializer
from confluent_kafka.serialization import MessageField, SerializationContext

logger = logging.getLogger(__name__)


async def publish_telemetry_stream(
    producer: AIOProducer,
    topic: str,
    publisher_id: str,
    publish_interval_s: float,
    avro_serializer: AvroSerializer,
) -> NoReturn:
    published_sample_count = 0
    try:
        sample_index = 0
        while True:
            timestamp_ns = time.time_ns()

            telemetry = {
                "publisher_id": publisher_id,
                "timestamp_ns": timestamp_ns,
                "temperature_c": float(sample_index),
                "humidity_pct": float(sample_index),
            }

            telemetry_payload_bytes = avro_serializer(
                telemetry,
                SerializationContext(topic, MessageField.VALUE),
            )

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

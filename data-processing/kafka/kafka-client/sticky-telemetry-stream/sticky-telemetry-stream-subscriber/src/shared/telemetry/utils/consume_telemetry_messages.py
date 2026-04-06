import logging

from confluent_kafka.aio import AIOConsumer
from confluent_kafka.schema_registry.avro import AvroDeserializer
from shared.telemetry.utils.process_telemetry_message import process_telemetry_message

logger = logging.getLogger(__name__)


async def consume_telemetry_messages(
    consumer: AIOConsumer,
    subscriber_id: str,
    poll_timeout_s: float,
    avro_deserializer: AvroDeserializer,
) -> None:
    while True:
        message = await consumer.poll(timeout=poll_timeout_s)
        if message is None:
            continue
        if message.error():
            logger.error(
                f"Subscriber {subscriber_id}: consumer error: {message.error()}",
            )
            continue
        process_telemetry_message(message, subscriber_id, avro_deserializer)

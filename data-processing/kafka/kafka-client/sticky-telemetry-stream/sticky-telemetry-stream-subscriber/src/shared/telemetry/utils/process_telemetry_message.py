import logging

from confluent_kafka import Message
from confluent_kafka.schema_registry.avro import AvroDeserializer
from confluent_kafka.serialization import MessageField, SerializationContext

logger = logging.getLogger(__name__)


def process_telemetry_message(
    message: Message,
    subscriber_id: str,
    avro_deserializer: AvroDeserializer,
) -> None:
    try:
        key_bytes = message.key()
        publisher_id = key_bytes.decode() if key_bytes else "unknown"
        telemetry = avro_deserializer(
            message.value(),
            SerializationContext(message.topic(), MessageField.VALUE),
        )
        telemetry_log = {
            "publisher_id": publisher_id,
            "partition": message.partition(),
            "timestamp_ns": telemetry["timestamp_ns"],
            "temperature_c": telemetry["temperature_c"],
            "humidity_pct": telemetry["humidity_pct"],
        }
        logger.info(
            f"Subscriber {subscriber_id}: {telemetry_log}",
        )

    except Exception:
        logger.exception(
            f"Subscriber {subscriber_id}: error processing message from partition {message.partition()}",
        )

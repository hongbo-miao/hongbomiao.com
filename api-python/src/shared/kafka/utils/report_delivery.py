import logging

from confluent_kafka import cimpl

logger = logging.getLogger(__name__)


def report_delivery(err: cimpl.KafkaError, message: cimpl.Message) -> None:
    if err is not None:
        logger.error(f"Message delivery failed: {err}")
    else:
        logger.info(
            "Message delivery succeed.",
            extra={
                "topic": message.topic(),
                "partition": message.partition(),
                "value": message.value(),
            },
        )

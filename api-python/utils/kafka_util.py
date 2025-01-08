from confluent_kafka import cimpl
from utils.logger import logger


def delivery_report(err: cimpl.KafkaError, msg: cimpl.Message) -> None:
    if err is not None:
        logger.error(f"Message delivery failed: {err}")
    else:
        logger.info(
            "Message delivery succeed.",
            extra={
                "topic": msg.topic(),
                "partition": msg.partition(),
                "value": msg.value(),
            },
        )

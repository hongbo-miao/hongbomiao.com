import logging
import os

import pulsar
from pulsar.schema import AvroSchema
from shared.telemetry.utils.handle_send_result import handle_send_result
from shared.telemetry.utils.validate_telemetry_json import validate_telemetry_json
from sticky_telemetry_stream_schema.telemetry_record import TelemetryRecord

logger = logging.getLogger(__name__)

PULSAR_SERVICE_URL = os.environ.get("PULSAR_SERVICE_URL", "pulsar://localhost:6650")
INPUT_TOPIC = "persistent://public/default/sensor-telemetry-raw"
OUTPUT_TOPIC = "persistent://public/default/sensor-telemetry"
SUBSCRIPTION_NAME = "validate-telemetry-subscription"
RECEIVE_TIMEOUT_MS = 1000


def main() -> None:
    client = None
    producer = None
    try:
        logger.info(f"Connecting to Pulsar at {PULSAR_SERVICE_URL}")
        client = pulsar.Client(PULSAR_SERVICE_URL)

        consumer = client.subscribe(
            topic=INPUT_TOPIC,
            subscription_name=SUBSCRIPTION_NAME,
            consumer_type=pulsar.ConsumerType.Shared,
            initial_position=pulsar.InitialPosition.Earliest,
        )

        producer = client.create_producer(
            topic=OUTPUT_TOPIC,
            schema=AvroSchema(TelemetryRecord),
            producer_name="validate-telemetry-producer",
        )

        logger.info(
            f"Consuming from {INPUT_TOPIC}, producing validated Avro to {OUTPUT_TOPIC}",
        )

        while True:
            try:
                message = consumer.receive(timeout_millis=RECEIVE_TIMEOUT_MS)
            except pulsar.Timeout:
                continue
            except pulsar.PulsarException:
                logger.exception("Failed to receive message")
                continue

            try:
                raw_payload = message.data()
                telemetry_record = validate_telemetry_json(raw_payload)
                if telemetry_record is None:
                    logger.warning(
                        f"Dropping invalid message {message.message_id()}",
                    )
                    consumer.acknowledge(message)
                    continue

                producer.send_async(
                    telemetry_record,
                    partition_key=telemetry_record.publisher_id,
                    callback=handle_send_result,
                )
                consumer.acknowledge(message)
            except Exception:
                logger.exception(
                    f"Failed to process message {message.message_id()}",
                )
                consumer.negative_acknowledge(message)

    except Exception:
        logger.exception("Validate function error")
        raise
    finally:
        if producer:
            producer.flush()
        if client:
            client.close()
            logger.info("Pulsar client closed")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()

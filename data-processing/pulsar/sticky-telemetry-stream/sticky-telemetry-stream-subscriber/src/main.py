import logging
import os
from uuid import uuid4

import pulsar
from pulsar.schema import AvroSchema
from shared.telemetry.utils.consume_telemetry_messages import (
    consume_telemetry_messages,
)
from sticky_telemetry_stream_schema.telemetry_record import TelemetryRecord

logger = logging.getLogger(__name__)

PULSAR_SERVICE_URL = os.environ.get("PULSAR_SERVICE_URL", "pulsar://localhost:6650")
PULSAR_TOPIC = "persistent://public/default/sensor-telemetry-streams"
SUBSCRIPTION_NAME = "telemetry-subscriber-group"
RECEIVE_TIMEOUT_MS = 1000


def main() -> None:
    client = None
    try:
        subscriber_id = str(uuid4())[:8]
        logger.info(
            f"Subscriber {subscriber_id}: connecting to Pulsar at {PULSAR_SERVICE_URL}",
        )
        client = pulsar.Client(PULSAR_SERVICE_URL)
        consumer = client.subscribe(
            topic=PULSAR_TOPIC,
            subscription_name=SUBSCRIPTION_NAME,
            consumer_type=pulsar.ConsumerType.KeyShared,
            initial_position=pulsar.InitialPosition.Earliest,
            schema=AvroSchema(TelemetryRecord),
        )

        logger.info(
            f"Subscriber {subscriber_id}: joined subscription '{SUBSCRIPTION_NAME}' "
            f"with KeyShared consumer type",
        )

        consume_telemetry_messages(consumer, subscriber_id, RECEIVE_TIMEOUT_MS)

    except Exception:
        logger.exception("Subscriber error")
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

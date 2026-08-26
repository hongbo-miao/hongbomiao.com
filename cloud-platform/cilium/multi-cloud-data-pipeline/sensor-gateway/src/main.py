import logging
import os
from http.server import HTTPServer

import pulsar
from shared.telemetry.handlers.create_sensor_reading_handler import (
    create_sensor_reading_handler,
)

logger = logging.getLogger(__name__)

PULSAR_SERVICE_URL = os.environ.get("PULSAR_SERVICE_URL", "pulsar://localhost:6650")
PULSAR_TOPIC = os.environ.get("PULSAR_TOPIC", "persistent://telemetry/sensor/reading")
CLOUD_NAME = os.environ.get("CLOUD_NAME", "unknown")
HTTP_PORT = int(os.environ.get("HTTP_PORT", "8080"))


def main() -> None:
    logger.info(f"Connecting to Pulsar at {PULSAR_SERVICE_URL}")
    client = pulsar.Client(PULSAR_SERVICE_URL)
    producer = client.create_producer(PULSAR_TOPIC)

    handler = create_sensor_reading_handler(producer, CLOUD_NAME)
    server = HTTPServer(("0.0.0.0", HTTP_PORT), handler)  # noqa: S104
    logger.info(f"sensor-gateway ({CLOUD_NAME}) listening on :{HTTP_PORT}")
    try:
        server.serve_forever()
    finally:
        producer.flush()
        client.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()

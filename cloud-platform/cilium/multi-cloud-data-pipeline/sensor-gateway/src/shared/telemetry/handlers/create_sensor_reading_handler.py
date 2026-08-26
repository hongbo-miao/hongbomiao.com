import json
import logging
import time
from http.server import BaseHTTPRequestHandler

import pulsar
from pydantic import ValidationError
from shared.telemetry.schemas.telemetry_reading import TelemetryReading
from shared.telemetry.services.publish_reading import publish_reading

logger = logging.getLogger(__name__)


def create_sensor_reading_handler(
    producer: pulsar.Producer,
    cloud_name: str,
) -> type[BaseHTTPRequestHandler]:
    class SensorReadingHandler(BaseHTTPRequestHandler):
        def log_message(self, format_string: str, *args: object) -> None:
            logger.info(format_string, *args)

        def do_GET(self) -> None:
            if self.path != "/healthz":
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.end_headers()

        def do_POST(self) -> None:
            if self.path != "/readings":
                self.send_response(404)
                self.end_headers()
                return

            content_length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(content_length))
                body.setdefault("timestamp_ns", time.time_ns())
                reading = TelemetryReading(**body)
            except (json.JSONDecodeError, ValidationError) as error:
                logger.exception("Rejected malformed reading")
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(error)}).encode())
                return

            publish_reading(producer, reading)

            self.send_response(202)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {"cloud": cloud_name, "device_id": reading.device_id},
                ).encode(),
            )

    return SensorReadingHandler

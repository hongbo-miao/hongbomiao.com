import json
import logging

from sticky_telemetry_stream_schema.telemetry_record import TelemetryRecord

logger = logging.getLogger(__name__)


def validate_telemetry_json(raw_payload: bytes) -> TelemetryRecord | None:
    try:
        record = json.loads(raw_payload)
    except (json.JSONDecodeError, TypeError):
        logger.exception("Failed to parse JSON payload")
        return None

    publisher_id = record.get("publisher_id")
    if not isinstance(publisher_id, str) or not publisher_id:
        logger.error(f"Invalid or missing publisher_id: {publisher_id}")
        return None

    timestamp_ns = record.get("timestamp_ns")
    if not isinstance(timestamp_ns, int):
        logger.error(f"Invalid or missing timestamp_ns: {timestamp_ns}")
        return None

    temperature_c = record.get("temperature_c")
    if temperature_c is not None and not isinstance(temperature_c, (int, float)):
        logger.error(f"Invalid temperature_c: {temperature_c}")
        return None

    humidity_pct = record.get("humidity_pct")
    if humidity_pct is not None and not isinstance(humidity_pct, (int, float)):
        logger.error(f"Invalid humidity_pct: {humidity_pct}")
        return None

    return TelemetryRecord(
        publisher_id=publisher_id,
        timestamp_ns=timestamp_ns,
        temperature_c=float(temperature_c) if temperature_c is not None else None,
        humidity_pct=float(humidity_pct) if humidity_pct is not None else None,
    )

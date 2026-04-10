import logging

from google.protobuf.message import DecodeError
from shared.telemetry.proto.telemetry_record_pb2 import (
    TelemetryRecord as TelemetryRecordPb,
)
from sticky_telemetry_stream_schema.telemetry_record import TelemetryRecord

logger = logging.getLogger(__name__)


def validate_telemetry_protobuf(raw_payload: bytes) -> TelemetryRecord | None:
    try:
        pb_record = TelemetryRecordPb()
        pb_record.ParseFromString(raw_payload)
    except DecodeError:
        logger.exception("Failed to parse protobuf payload")
        return None

    publisher_id = pb_record.publisher_id
    if not publisher_id:
        logger.error(f"Invalid or missing publisher_id: {publisher_id!r}")
        return None

    timestamp_ns = pb_record.timestamp_ns
    if timestamp_ns == 0:
        logger.error(f"Invalid or missing timestamp_ns: {timestamp_ns}")
        return None

    temperature_c = (
        pb_record.temperature_c if pb_record.HasField("temperature_c") else None
    )
    humidity_pct = (
        pb_record.humidity_pct if pb_record.HasField("humidity_pct") else None
    )

    return TelemetryRecord(
        publisher_id=publisher_id,
        timestamp_ns=timestamp_ns,
        temperature_c=temperature_c,
        humidity_pct=humidity_pct,
    )

from pulsar.schema import Double, Long, Record, String


class TelemetryRecord(Record):
    publisher_id = String(required=True)
    timestamp_ns = Long(required=True)
    temperature_c = Double(required=False, default=None)
    humidity_pct = Double(required=False, default=None)

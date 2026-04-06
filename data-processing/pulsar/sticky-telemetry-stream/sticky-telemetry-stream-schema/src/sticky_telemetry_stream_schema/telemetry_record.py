from pulsar.schema import Double, Long, Record, String


class TelemetryRecord(Record):
    publisher_id = String()
    timestamp_ns = Long()
    temperature_c = Double(required=False, default=None)
    humidity_pct = Double(required=False, default=None)

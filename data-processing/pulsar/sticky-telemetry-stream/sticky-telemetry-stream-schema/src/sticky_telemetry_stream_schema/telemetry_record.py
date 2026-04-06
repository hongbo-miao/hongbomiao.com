from pulsar.schema import Double, Long, Record


class TelemetryRecord(Record):
    timestamp_ns = Long()
    temperature_c = Double(required=False, default=None)
    humidity_pct = Double(required=False, default=None)

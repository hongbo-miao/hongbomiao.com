from pulsar.schema import Array, Double, Record, String


class EntryRecord(Record):
    name = String()
    value = Double(required=False, default=None)


class TelemetryRecord(Record):
    timestamp = String()
    entries = Array(EntryRecord())

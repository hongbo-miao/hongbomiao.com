from pulsar.schema import Double, Integer, Long, Record, String


class TranscriptMetricRecord(Record):
    device_id = String()
    window_start_ms = Long()  # sliding window start, Unix epoch milliseconds
    window_end_ms = Long()  # sliding window end, Unix epoch milliseconds
    message_rate_per_sec = Double()  # transcript messages per second within the window
    message_count = Integer()  # number of transcript messages within the window
    avg_length_chars = Double()  # average transcript text length in characters
    # average time between consecutive messages; None when message_count < 2
    avg_gap_ms = Double(default=None)
    # shortest gap between consecutive messages; None when message_count < 2
    min_gap_ms = Double(default=None)
    # longest gap between consecutive messages; None when message_count < 2
    max_gap_ms = Double(default=None)

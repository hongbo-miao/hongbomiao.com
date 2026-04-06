import pulsar


class BatchResultRecord(pulsar.schema.Record):
    publisher_id = pulsar.schema.String()
    batch_index = pulsar.schema.Long()
    sample_count = pulsar.schema.Long()
    temperature_average = pulsar.schema.Double()
    first_timestamp_ns = pulsar.schema.Long()
    last_timestamp_ns = pulsar.schema.Long()

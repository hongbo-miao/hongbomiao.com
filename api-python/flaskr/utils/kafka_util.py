from confluent_kafka import cimpl


def delivery_report(err: cimpl.KafkaError, msg: cimpl.Message):
    if err is not None:
        print(f"Message delivery failed: {err}")
    else:
        print(
            "Message delivery succeed.",
            {"topic": msg.topic(), "partition": msg.partition(), "value": msg.value()},
        )

import json
import random
import time

from confluent_kafka import Producer
from flask import Blueprint, current_app
from flaskr.utils import kafka_util

motor_blueprint = Blueprint("motor_blueprint", __name__)


@motor_blueprint.post("/generate-motor-data")
def generate_motor_data() -> dict[str, bool]:
    producer = Producer(
        {"bootstrap.servers": current_app.config.get("KAFKA_BOOTSTRAP_SERVERS")}
    )
    for _ in range(5):
        data = {
            "timestamp": time.time() * 1000,
            "current": random.uniform(0, 10),
            "voltage": random.uniform(0, 20),
            "temperature": random.uniform(0, 50) + 25,
        }
        producer.poll(0)
        producer.produce(
            "hm.motor",
            json.dumps(data).encode("utf-8"),
            callback=kafka_util.delivery_report,
        )
        time.sleep(1)
    producer.flush()
    return {"body": True}

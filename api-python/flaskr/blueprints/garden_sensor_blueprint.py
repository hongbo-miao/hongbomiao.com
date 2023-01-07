import json
import random
import time

from confluent_kafka import Producer
from flask import Blueprint, current_app
from flaskr.utils import kafka_util

garden_sensor_blueprint = Blueprint("garden_sensor_blueprint", __name__)


@garden_sensor_blueprint.post("/generate-garden-sensor-data")
def generate_garden_sensor_data() -> dict[str, bool]:
    producer = Producer(
        {"bootstrap.servers": current_app.config.get("KAFKA_BOOTSTRAP_SERVERS")}
    )
    for _ in range(5):
        data = {
            "temperature": round(random.uniform(-10, 50), 1),
            "humidity": round(random.uniform(0, 100), 1),
            "wind": round(random.uniform(0, 10), 1),
            "soil": round(random.uniform(0, 100), 1),
        }
        producer.poll(0)
        producer.produce(
            "garden_sensor_data",
            json.dumps(data).encode("utf-8"),
            callback=kafka_util.delivery_report,
        )
        time.sleep(1)
    producer.flush()
    return {"body": True}

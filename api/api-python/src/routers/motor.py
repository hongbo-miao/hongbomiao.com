import asyncio
import json
import time
from random import SystemRandom
from typing import Annotated

from config import config
from confluent_kafka import Producer
from fastapi import APIRouter, Depends
from shared.kafka.utils.report_delivery import report_delivery

router = APIRouter()


def get_producer() -> Producer:
    producer = Producer({"bootstrap.servers": config.KAFKA_BOOTSTRAP_SERVERS})
    try:
        yield producer
    finally:
        producer.flush()


@router.post("/generate-motor-data", tags=["motor"])
async def generate_motor_data(
    producer: Annotated[Producer, Depends(get_producer)],
) -> dict[str, bool]:
    random_number_generator = SystemRandom()
    for _ in range(5):
        data = {
            "timestamp": time.time() * 1000,
            "current": random_number_generator.uniform(0, 10),
            "voltage": random_number_generator.uniform(0, 20),
            "temperature": random_number_generator.uniform(0, 50) + 25,
        }
        producer.poll(0)
        producer.produce(
            "hm.motor",
            json.dumps(data).encode("utf-8"),
            callback=report_delivery,
        )
        await asyncio.sleep(1)
    return {"body": True}

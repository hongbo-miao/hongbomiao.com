import asyncio
import json
import secrets
import time
from typing import Annotated

from config import Config
from confluent_kafka import Producer
from fastapi import APIRouter, Depends
from utils.kafka_util import delivery_report

router = APIRouter()
config = Config()


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
    for _ in range(5):
        data = {
            "timestamp": time.time() * 1000,
            "current": secrets.SystemRandom().uniform(0, 10),
            "voltage": secrets.SystemRandom().uniform(0, 20),
            "temperature": secrets.SystemRandom().uniform(0, 50) + 25,
        }
        producer.poll(0)
        producer.produce(
            "hm.motor",
            json.dumps(data).encode("utf-8"),
            callback=delivery_report,
        )
        await asyncio.sleep(1)
    return {"body": True}

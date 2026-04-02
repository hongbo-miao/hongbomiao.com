import nats
from nats.js.client import JetStreamContext
from shared.telemetry.utils.process_telemetry_message import process_telemetry_message


async def fetch_messages(
    consumer: JetStreamContext.PullSubscription,
    subscriber_id: str,
    fetch_batch_size: int,
) -> None:
    while True:
        try:
            messages = await consumer.fetch(batch=fetch_batch_size, timeout=5)
            for message in messages:
                await process_telemetry_message(message, subscriber_id)
        except nats.errors.TimeoutError:
            continue

import asyncio
import logging

from nats.js.kv import KeyValue

logger = logging.getLogger(__name__)


async def run_lease_heartbeat(
    key_value_store: KeyValue,
    partition_index: int,
    subscriber_id: str,
    initial_revision: int,
    heartbeat_interval_seconds: float,
) -> None:
    revision = initial_revision
    key = f"partition.{partition_index}"
    while True:
        await asyncio.sleep(heartbeat_interval_seconds)
        try:
            revision = await key_value_store.update(
                key,
                subscriber_id.encode(),
                revision,
            )
        except Exception:
            logger.exception(
                f"Subscriber {subscriber_id}: failed to renew lease for partition {partition_index}",
            )
            raise

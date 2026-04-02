import asyncio
import logging

from nats.js.kv import KeyValue
from shared.partition.services.claim_partition import claim_partition

logger = logging.getLogger(__name__)


async def wait_for_partition_claim(
    key_value_store: KeyValue,
    subscriber_id: str,
    total_partition_count: int,
    claim_retry_interval_s: float,
) -> tuple[int, int]:
    while True:
        try:
            claimed_partition_index, initial_revision = await claim_partition(
                key_value_store,
                subscriber_id,
                total_partition_count,
            )
        except RuntimeError:
            logger.warning(
                f"Subscriber {subscriber_id}: no available partitions, "
                f"retrying in {claim_retry_interval_s}s",
            )
            await asyncio.sleep(claim_retry_interval_s)
        else:
            return claimed_partition_index, initial_revision

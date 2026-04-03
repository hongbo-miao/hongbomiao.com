import logging

from nats.js.errors import KeyValueError, KeyWrongLastSequenceError
from nats.js.kv import KeyValue

logger = logging.getLogger(__name__)


async def claim_partition(
    key_value_store: KeyValue,
    subscriber_id: str,
    total_partition_count: int,
) -> tuple[int, int]:
    for partition_index in range(total_partition_count):
        key = f"partition.{partition_index}"
        try:
            revision = await key_value_store.create(key, subscriber_id.encode())
        except (KeyValueError, KeyWrongLastSequenceError):
            logger.debug(
                f"Subscriber {subscriber_id}: partition {partition_index} already claimed, skipping",
            )
            continue
        else:
            logger.info(
                f"Subscriber {subscriber_id}: claimed partition {partition_index}",
            )
            return partition_index, revision

    message = (
        f"Subscriber {subscriber_id}: no available partitions "
        f"(total: {total_partition_count})"
    )
    raise RuntimeError(message)

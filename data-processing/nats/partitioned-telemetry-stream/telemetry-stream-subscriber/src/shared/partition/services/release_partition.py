import logging

from nats.js.kv import KeyValue

logger = logging.getLogger(__name__)


async def release_partition(
    key_value_store: KeyValue,
    partition_index: int,
    subscriber_id: str,
) -> None:
    key = f"partition.{partition_index}"
    try:
        await key_value_store.delete(key)
        logger.info(
            f"Subscriber {subscriber_id}: released partition {partition_index}",
        )
    except Exception:
        logger.exception(
            f"Subscriber {subscriber_id}: failed to release partition {partition_index}",
        )

import asyncio
import logging
import os
from uuid import uuid4

import nats
from nats.js.api import KeyValueConfig
from shared.partition.services.release_partition import release_partition
from shared.partition.services.run_lease_heartbeat import run_lease_heartbeat
from shared.partition.services.wait_for_partition_claim import wait_for_partition_claim
from shared.telemetry.utils.fetch_messages import fetch_messages

logger = logging.getLogger(__name__)

NATS_URL = os.environ.get("NATS_URL", "nats://localhost:4222")
NATS_STREAM_NAME = "SENSOR_TELEMETRY_STREAMS"
TOTAL_PARTITION_COUNT = int(os.environ.get("TOTAL_PARTITION_COUNT", "2"))
LEASE_TTL_S = 2
HEARTBEAT_INTERVAL_S = 0.5
KV_BUCKET_NAME = "PARTITION_LEASES"
FETCH_BATCH_SIZE = 100
CLAIM_RETRY_INTERVAL_S = 0.1


async def main() -> None:
    nats_client = None
    claimed_partition_index = None
    key_value_store = None
    heartbeat_task = None
    try:
        subscriber_id = str(uuid4())[:8]
        logger.info(f"Subscriber {subscriber_id}: connecting to NATS at {NATS_URL}")
        nats_client = await nats.connect(NATS_URL)
        jetstream_context = nats_client.jetstream()

        key_value_store = await jetstream_context.create_key_value(
            config=KeyValueConfig(bucket=KV_BUCKET_NAME, ttl=LEASE_TTL_S),
        )
        claimed_partition_index, initial_revision = await wait_for_partition_claim(
            key_value_store,
            subscriber_id,
            TOTAL_PARTITION_COUNT,
            CLAIM_RETRY_INTERVAL_S,
        )

        heartbeat_task = asyncio.create_task(
            run_lease_heartbeat(
                key_value_store,
                claimed_partition_index,
                subscriber_id,
                initial_revision,
                HEARTBEAT_INTERVAL_S,
            ),
        )

        subject_filter = f"{NATS_STREAM_NAME}.*.{claimed_partition_index}"
        durable_name = f"telemetry_partition_{claimed_partition_index}"
        consumer = await jetstream_context.pull_subscribe(
            subject_filter,
            durable=durable_name,
        )
        logger.info(
            f"Subscriber {subscriber_id}: created pull consumer for '{subject_filter}' "
            f"with durable name '{durable_name}'",
        )

        await asyncio.gather(
            heartbeat_task,
            fetch_messages(consumer, subscriber_id, FETCH_BATCH_SIZE),
        )

    except Exception:
        logger.exception("Subscriber error")
        raise
    finally:
        if heartbeat_task:
            heartbeat_task.cancel()
        if key_value_store and claimed_partition_index is not None:
            await release_partition(
                key_value_store,
                claimed_partition_index,
                subscriber_id,
            )
        if nats_client:
            await nats_client.drain()
            logger.info("NATS connection closed")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    asyncio.run(main())

import asyncio
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path

import nats
from nats.aio.msg import Msg
from nats.js.api import RetentionPolicy, StorageType
from nats.js.client import JetStreamContext
from nats.js.errors import NotFoundError

logger = logging.getLogger(__name__)

NATS_URL = "nats://localhost:4222"
STREAM_NAME = "AUDIO_STREAMS"
SUBJECT_PREFIX = "audio.streams.flac"
SUBJECT_PATTERN = f"{SUBJECT_PREFIX}.>"
CONSUMER_NAME = "audio_flac_consumer"


async def ensure_stream_exists(jetstream_context: JetStreamContext) -> None:
    try:
        await jetstream_context.stream_info(STREAM_NAME)
        logger.info(f"Stream '{STREAM_NAME}' already exists")
    except NotFoundError:
        logger.info(f"Creating stream '{STREAM_NAME}'")
        await jetstream_context.add_stream(
            name=STREAM_NAME,
            subjects=[SUBJECT_PATTERN],
            retention=RetentionPolicy.LIMITS,
            storage=StorageType.FILE,
            max_age=86400,  # 24 hours in seconds
        )
        logger.info(f"Stream '{STREAM_NAME}' created successfully")


def create_message_handler(
    data_directory: Path,
) -> Callable[[Msg], Awaitable[None]]:
    async def handle_audio_message(message: Msg) -> None:
        try:
            subject_suffix = message.subject.removeprefix(f"{SUBJECT_PREFIX}.")
            flac_path = data_directory.joinpath(f"{subject_suffix}.flac")

            flac_path.write_bytes(message.data)
            await message.ack()

            logger.info(
                f"Saved '{flac_path.name}' from subject '{message.subject}' "
                f"(size: {len(message.data)} bytes)",
            )
        except Exception:
            logger.exception(f"Error processing message from '{message.subject}'")
            await message.nak()

    return handle_audio_message


async def main() -> None:
    nats_client = None
    try:
        logger.info(f"Connecting to NATS at {NATS_URL}")
        nats_client = await nats.connect(NATS_URL)
        jetstream_context = nats_client.jetstream()

        await ensure_stream_exists(jetstream_context)

        data_directory = Path("data")
        data_directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving FLAC files to: {data_directory}")

        message_handler = create_message_handler(data_directory)

        await jetstream_context.subscribe(
            SUBJECT_PATTERN,
            durable=CONSUMER_NAME,
            cb=message_handler,
            manual_ack=True,
        )
        logger.info(
            f"Subscribed to '{SUBJECT_PATTERN}' with consumer '{CONSUMER_NAME}'",
        )

        await asyncio.Event().wait()

    except Exception:
        logger.exception("Subscriber error")
        raise
    finally:
        if nats_client:
            await nats_client.drain()
            logger.info("NATS connection closed")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    asyncio.run(main())

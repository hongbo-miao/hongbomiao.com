import asyncio
import logging
from collections.abc import Awaitable, Callable

import nats
from anyio import Path
from nats.aio.msg import Msg

logger = logging.getLogger(__name__)

NATS_URL = "nats://localhost:4222"
STREAM_NAME = "AUDIO_STREAMS"
SUBJECT_PREFIX = "AUDIO_STREAMS"
SUBJECT_PATTERN = f"{SUBJECT_PREFIX}.>"
CONSUMER_NAME = "audio_consumer"


def create_message_handler(
    data_directory: Path,
) -> Callable[[Msg], Awaitable[None]]:
    async def handle_audio_message(message: Msg) -> None:
        try:
            subject_suffix = message.subject.removeprefix(f"{SUBJECT_PREFIX}.")
            audio_path = data_directory / f"{subject_suffix}.flac"

            await audio_path.write_bytes(message.data)
            await message.ack()

            logger.info(
                f"Saved '{audio_path.name}' from subject '{message.subject}' "
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

        data_directory = Path("data")
        await data_directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving audio files to: {data_directory}")

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

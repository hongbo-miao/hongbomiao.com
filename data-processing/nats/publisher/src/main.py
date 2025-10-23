import asyncio
import logging
from pathlib import Path

import nats
from nats.js.api import RetentionPolicy, StorageType
from nats.js.client import JetStreamContext
from nats.js.errors import NotFoundError

logger = logging.getLogger(__name__)

NATS_URL = "nats://localhost:4222"
STREAM_NAME = "AUDIO_STREAMS"
SUBJECT_PREFIX = "audio.streams.flac"
SUBJECT_PATTERN = f"{SUBJECT_PREFIX}.>"


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


async def publish_single_flac(
    jetstream_context: JetStreamContext,
    flac_path: Path,
) -> bool:
    try:
        flac_bytes = flac_path.read_bytes()
        subject = f"{SUBJECT_PREFIX}.{flac_path.stem}"

        ack = await jetstream_context.publish(subject, flac_bytes)
    except Exception:
        logger.exception(f"Failed to publish '{flac_path.name}'")
        return False
    else:
        logger.info(
            f"Published '{flac_path.name}' to subject '{subject}' "
            f"(seq: {ack.seq}, stream: {ack.stream})",
        )
        return True


async def publish_flac_files(
    jetstream_context: JetStreamContext,
    data_directory: Path,
) -> int:
    flac_paths: list[Path] = sorted(data_directory.glob("*.flac"))

    if not flac_paths:
        logger.warning(f"No FLAC files found in {data_directory}")
        return 0

    published_count = 0
    for flac_path in flac_paths:
        published_count += int(
            await publish_single_flac(
                jetstream_context=jetstream_context,
                flac_path=flac_path,
            ),
        )

    return published_count


async def run_publisher() -> None:
    nats_client = None
    try:
        logger.info(f"Connecting to NATS at {NATS_URL}")
        nats_client = await nats.connect(NATS_URL)
        jetstream_context = nats_client.jetstream()

        await ensure_stream_exists(jetstream_context)

        data_directory = Path("data")
        published_count = await publish_flac_files(jetstream_context, data_directory)

        logger.info(f"Successfully published {published_count} FLAC files")

    except Exception:
        logger.exception("Publisher error")
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
    asyncio.run(run_publisher())

import asyncio
import logging
from uuid import uuid4

import nats
from anyio import Path
from nats.js.client import JetStreamContext

logger = logging.getLogger(__name__)

NATS_URL = "nats://localhost:4222"
SUBJECT_PREFIX = "AUDIO_STREAMS"


async def publish_single_flac(
    jetstream_context: JetStreamContext,
    audio_path: Path,
) -> bool:
    try:
        flac_bytes = await audio_path.read_bytes()
        subject = f"{SUBJECT_PREFIX}.{audio_path.stem}"
        ack = await jetstream_context.publish(
            subject,
            flac_bytes,
            headers={
                "Nats-Msg-Id": str(uuid4()),
            },
        )
    except Exception:
        logger.exception(f"Failed to publish '{audio_path.name}'")
        return False
    else:
        logger.info(
            f"Published '{audio_path.name}' to subject '{subject}' "
            f"(seq: {ack.seq}, stream: {ack.stream})",
        )
        return True


async def publish_flac_files(
    jetstream_context: JetStreamContext,
    data_directory: Path,
) -> int:
    audio_paths: list[Path] = sorted(
        [path async for path in data_directory.glob("*.flac")],
    )

    if not audio_paths:
        logger.warning(f"No FLAC files found in {data_directory}")
        return 0

    published_count = 0
    for audio_path in audio_paths:
        published_count += int(
            await publish_single_flac(
                jetstream_context=jetstream_context,
                audio_path=audio_path,
            ),
        )

    return published_count


async def main() -> None:
    nats_client = None
    try:
        logger.info(f"Connecting to NATS at {NATS_URL}")
        nats_client = await nats.connect(NATS_URL)
        jetstream_context = nats_client.jetstream()

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
    asyncio.run(main())

import asyncio
import logging
from typing import cast

import nats
from nats.js.client import JetStreamContext

logger = logging.getLogger(__name__)

NATS_URL = "nats://localhost:4222"
FIRE_STREAM_METADATA = {
    "identifier": "lincoln_fire",
    "name": "Lincoln Fire",
    # https://www.broadcastify.com/webPlayer/14395
    "stream_url": "https://listen.broadcastify.com/qmydkwhnjbzf80t.mp3",
    "location": "Lincoln, NE",
}
SUBJECT_PREFIX = "FIRE_AUDIO_STREAMS"
STREAM_SUBJECT = f"{SUBJECT_PREFIX}.{FIRE_STREAM_METADATA['identifier']}"
PCM_CHUNK_SIZE_BYTES = int(16000 * 2 * 1 * (200 / 1000))  # 200ms at 16kHz mono s16le


async def publish_audio_stream(
    jetstream_context: JetStreamContext,
    stream_url: str,
    subject: str,
) -> int:
    def validate_ffmpeg_stdout(process: asyncio.subprocess.Process) -> None:
        if process.stdout is None:
            message = "FFmpeg stdout is not available"
            raise RuntimeError(message)

    published_chunk_count = 0
    ffmpeg_process = None
    try:
        # Start FFmpeg process to decode MP3 to raw PCM
        ffmpeg_process = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-i",
            stream_url,
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-loglevel",
            "error",
            "-",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        logger.info(
            f"Streaming '{FIRE_STREAM_METADATA['name']}' "
            f"from {FIRE_STREAM_METADATA['location']}",
        )

        validate_ffmpeg_stdout(ffmpeg_process)

        # Type narrowing: validated by validate_ffmpeg_stdout
        stdout = cast("asyncio.StreamReader", ffmpeg_process.stdout)

        # Read PCM chunks from FFmpeg stdout
        while True:
            pcm_chunk = await stdout.read(PCM_CHUNK_SIZE_BYTES)
            if not pcm_chunk:
                break

            published_chunk_count += 1
            headers = {
                "Content-Type": "audio/pcm",
                "X-Stream-Name": FIRE_STREAM_METADATA["name"],
                "X-Stream-Location": FIRE_STREAM_METADATA["location"],
                "X-Chunk-Sequence-Number": str(published_chunk_count),
            }
            publish_acknowledgement = await jetstream_context.publish(
                subject,
                pcm_chunk,
                headers=headers,
            )

            if published_chunk_count % 50 == 0:
                logger.info(
                    f"Published {published_chunk_count} audio chunks "
                    f"(latest seq: {publish_acknowledgement.seq})",
                )

    except asyncio.CancelledError:
        logger.info("Audio streaming cancelled")
        raise
    except Exception:
        logger.exception("Failed to publish audio stream")
        raise
    finally:
        if ffmpeg_process:
            ffmpeg_process.terminate()
            try:
                await asyncio.wait_for(ffmpeg_process.wait(), timeout=5)
            except asyncio.TimeoutError:
                ffmpeg_process.kill()

    return published_chunk_count


async def main() -> None:
    nats_client = None
    try:
        logger.info(f"Connecting to NATS at {NATS_URL}")
        nats_client = await nats.connect(NATS_URL)
        jetstream_context = nats_client.jetstream()

        published_chunk_count = await publish_audio_stream(
            jetstream_context=jetstream_context,
            stream_url=FIRE_STREAM_METADATA["stream_url"],
            subject=STREAM_SUBJECT,
        )

        logger.info(
            f"Audio streaming completed with {published_chunk_count} chunks published",
        )

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

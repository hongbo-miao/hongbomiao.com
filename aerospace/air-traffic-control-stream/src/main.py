import asyncio
import contextlib
import logging
from collections.abc import AsyncGenerator
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

LIVEATC_BASE_URL = "http://d.liveatc.net"
STREAMS: dict[str, dict[str, str]] = {
    "kjfk_twr": {
        "name": "JFK Airport - Tower",
        "stream_path": "kjfk_twr",
        "location": "New York, NY",
    },
    "klax_twr": {
        "name": "LAX Airport - Tower",
        "stream_path": "klax_twr",
        "location": "Los Angeles, CA",
    },
    "ksfo_twr": {
        "name": "San Francisco International - Tower",
        "stream_path": "ksfo_twr",
        "location": "San Francisco, CA",
    },
}

logger = logging.getLogger(__name__)
app = FastAPI()


class StreamManager:
    def __init__(self) -> None:
        # Active streams: stream_id -> set of connected websockets
        self.active_streams: dict[str, set[WebSocket]] = {}
        # Stream tasks: stream_id -> asyncio.Task
        self.stream_tasks: dict[str, asyncio.Task] = {}
        # Stream queues: stream_id -> asyncio.Queue for broadcasting
        self.stream_queues: dict[str, asyncio.Queue] = {}

    async def add_client(self, stream_id: str, websocket: WebSocket) -> None:
        if stream_id not in self.active_streams:
            self.active_streams[stream_id] = set()
            self.stream_queues[stream_id] = asyncio.Queue()
            # Start the stream decoder task
            self.stream_tasks[stream_id] = asyncio.create_task(
                self._decode_stream(stream_id),
            )
            logger.info(f"Started decoding stream: {stream_id}")

        self.active_streams[stream_id].add(websocket)
        logger.info(
            f"Client connected to {stream_id}. Total clients: {len(self.active_streams[stream_id])}",
        )

    async def remove_client(self, stream_id: str, websocket: WebSocket) -> None:
        """Remove a client from a stream."""
        if stream_id in self.active_streams:
            self.active_streams[stream_id].discard(websocket)

            # If no more clients, stop the stream
            if not self.active_streams[stream_id]:
                logger.info(f"No more clients for {stream_id}, stopping stream")

                # Cancel the decoder task
                if stream_id in self.stream_tasks:
                    self.stream_tasks[stream_id].cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await self.stream_tasks[stream_id]
                    del self.stream_tasks[stream_id]

                # Clean up
                del self.active_streams[stream_id]
                if stream_id in self.stream_queues:
                    del self.stream_queues[stream_id]
            else:
                logger.info(
                    f"Client disconnected from {stream_id}. Remaining clients: {len(self.active_streams[stream_id])}",
                )

    async def _decode_stream(self, stream_id: str) -> None:
        url = f"{LIVEATC_BASE_URL}/{STREAMS[stream_id]['stream_path']}"

        try:
            async for chunk in self._ffmpeg_pcm_stream(url):
                if stream_id not in self.active_streams:
                    break

                # Broadcast to all connected clients
                disconnected_clients = set()
                for websocket in self.active_streams[stream_id].copy():
                    try:
                        await websocket.send_bytes(chunk)
                    except Exception:
                        logger.exception("Failed to send to client.")
                        disconnected_clients.add(websocket)

                # Remove disconnected clients
                for websocket in disconnected_clients:
                    await self.remove_client(stream_id, websocket)

        except asyncio.CancelledError:
            logger.info(f"Stream decoder cancelled for {stream_id}")
            raise
        except Exception:
            logger.exception(f"Error in stream decoder for {stream_id}.")

    async def _ffmpeg_pcm_stream(self, url: str) -> AsyncGenerator[bytes]:
        # Run ffmpeg as an async subprocess to decode MP3 stream to PCM16 s16le @ 16kHz mono. Yields small PCM chunks.
        process = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "quiet",
            "-i",
            url,
            "-ac",
            "1",  # mono
            "-ar",
            "16000",  # 16,000 Hz
            "-f",
            "s16le",  # PCM 16-bit little endian (2 bytes)
            "pipe:1",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        if process.stdout is None:
            msg = "Failed to create subprocess stdout pipe"
            raise RuntimeError(msg)

        try:
            while True:
                # 16,000 samples/sec x 2 bytes/sample x 1 channel x 0.1 sec = 3,200 bytes
                chunk = await process.stdout.read(3200)  # 0.1 sec of audio
                if not chunk:
                    break
                yield chunk
        finally:
            process.kill()
            await process.wait()


# Global stream manager
stream_manager = StreamManager()


@app.websocket("/ws/air-traffic-control-stream")
async def websocket_endpoint(websocket: WebSocket, stream_id: str) -> None:
    # Client connects and receives PCM audio chunks.
    await websocket.accept()

    if stream_id not in STREAMS:
        await websocket.send_text(f"Unknown {stream_id = }")
        await websocket.close()
        return

    try:
        await stream_manager.add_client(stream_id, websocket)

        # Keep the connection alive and wait for disconnect
        while True:
            try:
                # Wait for any message from client (keepalive, etc.)
                await websocket.receive_text()
            except WebSocketDisconnect:
                break
            except Exception:
                logger.exception("WebSocket error.")
                break

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception:
        logger.exception("Error in websocket endpoint.")
    finally:
        await stream_manager.remove_client(stream_id, websocket)


@app.get("/api/air-traffic-control-streams")
def get_streams() -> dict[str, dict[str, dict[str, str]]]:
    return {"streams": STREAMS}


@app.get("/api/air-traffic-control-stream-status")
def get_stream_status() -> dict[str, dict[str, dict[str, int | bool]]]:
    # Get current stream status for debugging
    status = {}
    for stream_id in stream_manager.active_streams:
        status[stream_id] = {
            "active": True,
            "client_count": len(stream_manager.active_streams[stream_id]),
            "has_decoder": stream_id in stream_manager.stream_tasks,
        }
    return {"status": status}


@app.get("/")
def index() -> HTMLResponse:
    html_file = Path(__file__).parent / "templates" / "index.html"
    html_content = html_file.read_text(encoding="utf-8")
    return HTMLResponse(html_content)

import logging
import struct
import tempfile
import threading
import time
from pathlib import Path
from typing import BinaryIO

import httpx
import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
import webrtcvad

logger = logging.getLogger(__name__)

SAMPLE_RATE: int = 16000
FRAME_DURATION: int = 30  # ms
API_URL: str = "http://localhost:34796/v1/audio/transcriptions"
MODEL_NAME: str = "Systran/faster-distil-whisper-medium.en"


class VadProcessor:
    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        frame_duration: int = FRAME_DURATION,
    ) -> None:
        self.sample_rate: int = sample_rate
        self.frame_size: int = int(sample_rate * frame_duration / 1000)
        self.vad: webrtcvad.Vad = webrtcvad.Vad(1)  # aggressiveness level 1

        # Buffers
        self.audio_buffer: list[np.int16] = []
        self.frame_buffer: list[np.int16] = []
        self.silence_frames: int = 0
        self.speech_frames_count: int = 0
        self.sentence_count: int = 0

        # Thresholds
        self.min_speech_frames: int = 5
        self.max_silence_frames: int = int(1000 / frame_duration)  # 1 second
        self.min_audio_length: int = int(0.3 * sample_rate)

    def audio_callback(
        self,
        indata: np.ndarray,
        frames: int,  # noqa: ARG002
        time_info: dict,  # noqa: ARG002
        status: sd.CallbackFlags | None,
    ) -> None:
        if status:
            logger.warning(f"Audio status: {status}")

        # Convert to mono int16
        chunk: np.ndarray = indata[:, 0] if len(indata.shape) > 1 else indata.flatten()
        chunk_int16: np.ndarray = (chunk * 32767).astype(np.int16)

        self.frame_buffer.extend(chunk_int16)
        self.audio_buffer.extend(chunk_int16)

        # Process complete frames
        while len(self.frame_buffer) >= self.frame_size:
            frame: np.ndarray = np.array(self.frame_buffer[: self.frame_size])
            self.frame_buffer = self.frame_buffer[self.frame_size :]

            # VAD check
            frame_bytes: bytes = struct.pack(f"{len(frame)}h", *frame)
            try:
                is_speech: bool = self.vad.is_speech(frame_bytes, self.sample_rate)
                if is_speech:
                    self.silence_frames = 0
                    self.speech_frames_count += 1
                else:
                    self.silence_frames += 1

                # Check if sentence is complete
                if self._is_complete():
                    self._process_audio()

            except Exception:
                logger.exception("VAD error.")

    def _is_complete(self) -> bool:
        audio_length: int = len(self.audio_buffer)
        duration: float = audio_length / self.sample_rate

        # Basic conditions
        if (
            audio_length < self.min_audio_length
            or self.speech_frames_count < self.min_speech_frames
        ):
            return False

        # Sentence complete conditions
        return (
            (self.silence_frames >= self.max_silence_frames and duration >= 0.5)
            or duration > 10.0  # Prevent infinite buffering
        )

    def _process_audio(self) -> None:
        if not self.audio_buffer:
            return

        audio_data: np.ndarray = np.array(self.audio_buffer, dtype=np.int16)
        self.sentence_count += 1

        # Reset buffers
        self.audio_buffer = []
        self.frame_buffer = []
        self.silence_frames = 0
        self.speech_frames_count = 0

        logger.info(f"🎯 Sentence {self.sentence_count} detected!")

        # Process in separate thread
        thread: threading.Thread = threading.Thread(
            target=self._transcribe,
            args=(audio_data, self.sentence_count),
            daemon=True,
        )
        thread.start()

    def _transcribe(self, audio_data: np.ndarray, sentence_num: int) -> None:
        try:
            duration: float = len(audio_data) / self.sample_rate
            logger.info(f"Processing sentence {sentence_num} ({duration:.2f}s)...")

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                wav.write(tmp_file.name, self.sample_rate, audio_data)
                result: str = self._send_to_api(tmp_file.name)
                logger.info(f"🎤 Sentence {sentence_num}: {result}")

        except Exception:
            logger.exception(f"Error processing sentence {sentence_num}.")

    def _send_to_api(self, file_path: str) -> str:
        with Path(file_path).open("rb") as f:
            files: dict[str, tuple[str, BinaryIO, str]] = {
                "file": (Path(file_path).name, f, "audio/wav"),
            }
            data: dict[str, str] = {"model": MODEL_NAME}
            with httpx.Client(timeout=30.0) as client:
                response: httpx.Response = client.post(API_URL, data=data, files=files)
                response.raise_for_status()
                transcription: str = response.json()["text"]
                return transcription

    def start_recording(self) -> None:
        chunk_size: int = int(self.sample_rate * 0.1)  # 100ms chunks
        logger.info("Starting recording. Speak naturally, press Ctrl+C to stop.")
        logger.info("Parameters:")
        logger.info(f"  - Min speech frames: {self.min_speech_frames}")
        logger.info(
            f"  - Max silence frames: {self.max_silence_frames} ({self.max_silence_frames * FRAME_DURATION / 1000:.1f}s)",
        )
        logger.info(
            f"  - Min audio length: {self.min_audio_length / self.sample_rate:.1f}s",
        )

        try:
            with sd.InputStream(
                callback=self.audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=chunk_size,
                dtype=np.float32,
            ):
                while True:
                    time.sleep(0.1)

        except KeyboardInterrupt:
            logger.info("Stopping...")
            # Process any remaining audio
            if len(self.audio_buffer) > self.min_audio_length:
                logger.info("Processing final audio segment...")
                self._process_audio()


def main() -> None:
    processor: VadProcessor = VadProcessor()
    processor.start_recording()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()

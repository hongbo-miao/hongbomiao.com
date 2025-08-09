import logging
import tempfile
import threading
import time
from pathlib import Path
from typing import BinaryIO

import httpx
import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
import torch
from silero_vad import load_silero_vad

logger = logging.getLogger(__name__)

SAMPLE_RATE: int = 16000
SPEACHES_API_URL: str = "http://localhost:34796/v1/audio/transcriptions"
MODEL_NAME: str = "Systran/faster-distil-whisper-medium.en"


class VadProcessor:
    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self.sample_rate: int = sample_rate

        # Load Silero VAD model
        self.vad_model = load_silero_vad()

        # Silero VAD expects exactly these sample counts
        if sample_rate == 16000:
            self.vad_window_size = 512  # 32ms at 16kHz
        elif sample_rate == 8000:
            self.vad_window_size = 256  # 32ms at 8kHz
        else:
            msg = f"Unsupported sample rate: {sample_rate}. Use 8000 or 16000."
            raise ValueError(
                msg,
            )

        # Buffers
        self.audio_buffer: list[np.float32] = []
        self.vad_buffer: list[np.float32] = []
        self.silence_windows: int = 0
        self.speech_windows_count: int = 0
        self.sentence_count: int = 0

        # Thresholds (based on 32ms windows)
        self.min_speech_windows: int = 16  # ~500ms of speech (16 * 32ms)
        self.max_silence_windows: int = 62  # ~2 seconds of silence (62 * 32ms)
        self.min_audio_length: int = int(0.3 * sample_rate)
        self.speech_threshold: float = 0.5  # VAD threshold

    def audio_callback(
        self,
        indata: np.ndarray,
        frames: int,  # noqa: ARG002
        time_info: dict,  # noqa: ARG002
        status: sd.CallbackFlags | None,
    ) -> None:
        if status:
            logger.warning(f"Audio status: {status}")

        # Convert to mono float32
        chunk: np.ndarray = indata[:, 0] if len(indata.shape) > 1 else indata.flatten()
        chunk_float32: np.ndarray = chunk.astype(np.float32)

        # Add to both buffers
        self.vad_buffer.extend(chunk_float32)
        self.audio_buffer.extend(chunk_float32)

        # Process VAD windows when we have enough data
        while len(self.vad_buffer) >= self.vad_window_size:
            vad_window: np.ndarray = np.array(self.vad_buffer[: self.vad_window_size])
            self.vad_buffer = self.vad_buffer[self.vad_window_size :]

            # Perform VAD check
            try:
                is_speech = self._check_speech(vad_window)

                if is_speech:
                    self.silence_windows = 0
                    self.speech_windows_count += 1
                else:
                    self.silence_windows += 1

                # Check if sentence is complete
                if self._is_complete():
                    self._process_audio()

            except Exception:
                logger.exception("VAD error.")

    def _check_speech(self, audio_window: np.ndarray) -> bool:
        try:
            # Ensure exact window size
            if len(audio_window) != self.vad_window_size:
                logger.warning(
                    f"Window size mismatch: {len(audio_window)} != {self.vad_window_size}",
                )
                return False

            # Convert to tensor
            audio_tensor = torch.from_numpy(audio_window).float()

            # Get speech probability
            speech_prob = self.vad_model(audio_tensor, self.sample_rate).item()
        except Exception:
            logger.exception("Speech detection error.")
            return False
        else:
            return speech_prob > self.speech_threshold

    def _is_complete(self) -> bool:
        audio_length: int = len(self.audio_buffer)
        duration: float = audio_length / self.sample_rate

        # Basic conditions
        if (
            audio_length < self.min_audio_length
            or self.speech_windows_count < self.min_speech_windows
        ):
            return False

        # Sentence complete conditions
        return (
            (self.silence_windows >= self.max_silence_windows and duration >= 0.5)
            or duration > 10.0  # Prevent infinite buffering
        )

    def _process_audio(self) -> None:
        if not self.audio_buffer:
            return

        # Convert float32 to int16 for WAV file
        audio_data: np.ndarray = np.array(self.audio_buffer, dtype=np.float32)
        audio_int16: np.ndarray = (audio_data * 32767).astype(np.int16)

        self.sentence_count += 1

        # Reset buffers and counters
        self.audio_buffer = []
        self.vad_buffer = []
        self.silence_windows = 0
        self.speech_windows_count = 0

        logger.info(f"🎯 Sentence {self.sentence_count} detected!")

        # Process in separate thread
        thread: threading.Thread = threading.Thread(
            target=self._transcribe,
            args=(audio_int16, self.sentence_count),
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
                response: httpx.Response = client.post(
                    SPEACHES_API_URL,
                    data=data,
                    files=files,
                )
                response.raise_for_status()
                transcription: str = response.json()["text"]
                return transcription

    def start_recording(self) -> None:
        chunk_size: int = int(self.sample_rate * 0.1)  # 100ms chunks
        logger.info("Starting recording. Speak naturally, press Ctrl+C to stop.")
        logger.info("Parameters:")
        logger.info(
            f"  - VAD window size: {self.vad_window_size} samples ({self.vad_window_size / self.sample_rate * 1000:.1f}ms)",
        )
        logger.info(
            f"  - Min speech windows: {self.min_speech_windows} (~{self.min_speech_windows * self.vad_window_size / self.sample_rate:.1f}s)",
        )
        logger.info(
            f"  - Max silence windows: {self.max_silence_windows} (~{self.max_silence_windows * self.vad_window_size / self.sample_rate:.1f}s)",
        )
        logger.info(
            f"  - Min audio length: {self.min_audio_length / self.sample_rate:.1f}s",
        )
        logger.info(f"  - Speech threshold: {self.speech_threshold}")

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

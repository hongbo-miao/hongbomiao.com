import logging
import tempfile
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd

logger = logging.getLogger(__name__)

DURATION: int = 3  # seconds
SAMPLE_RATE: int = 16000
API_URL: str = "http://localhost:34796/v1/audio/transcriptions"
MODEL_NAME: str = "Systran/faster-distil-whisper-medium.en"


def record_audio(duration: int, sample_rate: int) -> np.ndarray:
    logger.info(f"Recording for {duration} seconds...")
    recording: np.ndarray = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
    )
    sd.wait()
    logger.info("Recording complete.")
    return recording


def save_to_wav(data: np.ndarray, sample_rate: int, file_path: Path) -> None:
    wav.write(str(file_path), sample_rate, data)
    logger.info(f"Audio saved to {file_path}")


def transcribe_audio(file_path: Path) -> str:
    logger.info("Sending audio...")
    with file_path.open("rb") as audio_file:
        files: dict[str, tuple[str, Any, str]] = {
            "file": (file_path.name, audio_file, "audio/wav"),
        }
        data: dict[str, str] = {"model": MODEL_NAME}
        with httpx.Client(timeout=30.0) as client:
            response: httpx.Response = client.post(
                API_URL,
                data=data,
                files=files,
            )
    response.raise_for_status()
    transcription: str = response.json()["text"]
    logger.info("Transcription received.")
    return transcription


def main() -> None:
    recording: np.ndarray = record_audio(DURATION, SAMPLE_RATE)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path: Path = Path(tmpdir) / "recording.wav"
        save_to_wav(recording, SAMPLE_RATE, tmp_path)
        result: str = transcribe_audio(tmp_path)
    logger.info(f"Transcription result: {result}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()

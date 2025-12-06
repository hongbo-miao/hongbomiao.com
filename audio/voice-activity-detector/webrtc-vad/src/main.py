import logging
import struct
import time
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wav
import scipy.signal
import webrtcvad

logger = logging.getLogger(__name__)

SAMPLE_RATE_HZ: int = 16000
VAD_WINDOW_DURATION_MS: int = 30
MIN_SPEECH_WINDOW_COUNT: int = 5
MAX_SILENCE_WINDOW_COUNT: int = 31  # ~1 second of silence
MIN_AUDIO_LENGTH_S: float = 0.3
MIN_SEGMENT_DURATION_S: float = 0.5
MAX_SEGMENT_DURATION_S: float = 10.0
AUDIO_FILE_PATH: Path = Path("data/audio.wav")


class VadState:
    def __init__(self) -> None:
        self.audio_buffer: list[np.ndarray] = []
        self.silence_window_count: int = 0
        self.speech_window_count: int = 0
        self.segment_count: int = 0


class VadProcessor:
    @staticmethod
    def is_segment_complete(
        state: VadState,
        sample_rate: int,
        min_audio_length: int,
        min_speech_window_count: int,
        max_silence_window_count: int,
    ) -> bool:
        audio_length: int = sum(chunk.size for chunk in state.audio_buffer)
        duration: float = audio_length / sample_rate

        if (
            audio_length < min_audio_length
            or state.speech_window_count < min_speech_window_count
        ):
            return False

        return (
            state.silence_window_count >= max_silence_window_count
            and duration >= MIN_SEGMENT_DURATION_S
        ) or duration > MAX_SEGMENT_DURATION_S

    @staticmethod
    def process_segment(state: VadState, sample_rate: int) -> None:
        if not state.audio_buffer:
            return

        audio_data: np.ndarray = np.concatenate(state.audio_buffer)
        state.segment_count += 1

        # Reset buffers
        state.audio_buffer = []
        state.silence_window_count = 0
        state.speech_window_count = 0

        duration: float = len(audio_data) / sample_rate
        logger.info(f"Segment {state.segment_count} detected ({duration:.2f}s)")

    @staticmethod
    def process_audio_file(
        audio_file_path: Path,
        vad: webrtcvad.Vad,
        sample_rate: int = SAMPLE_RATE_HZ,
        vad_window_duration: int = VAD_WINDOW_DURATION_MS,
        min_speech_window_count: int = MIN_SPEECH_WINDOW_COUNT,
        max_silence_window_count: int = MAX_SILENCE_WINDOW_COUNT,
        min_audio_length_s: float = MIN_AUDIO_LENGTH_S,
    ) -> None:
        vad_window_size: int = int(sample_rate * vad_window_duration / 1000)
        min_audio_length: int = int(min_audio_length_s * sample_rate)
        state: VadState = VadState()

        logger.info(f"Processing audio file: {audio_file_path}")
        logger.info("Parameters:")
        logger.info(
            f"  - VAD window size: {vad_window_size} samples ({vad_window_duration:.1f}ms)",
        )
        logger.info(f"  - Min speech window count: {min_speech_window_count}")
        logger.info(
            f"  - Max silence window count: {max_silence_window_count} ({max_silence_window_count * vad_window_duration / 1000:.1f}s)",
        )
        logger.info(
            f"  - Min audio length: {min_audio_length / sample_rate:.1f}s",
        )

        file_sample_rate, audio_data = wav.read(audio_file_path)
        if file_sample_rate != sample_rate:
            logger.info(
                f"Resampling audio from {file_sample_rate} Hz to {sample_rate} Hz",
            )
            sample_count: int = int(
                len(audio_data) * sample_rate / file_sample_rate,
            )
            audio_data = scipy.signal.resample(audio_data, sample_count)

        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]

        # Convert to int16 if needed
        if audio_data.dtype in (np.float32, np.float64):
            audio_data = (audio_data * 32767).astype(np.int16)
        elif audio_data.dtype != np.int16:
            audio_data = audio_data.astype(np.int16)

        total_duration: float = len(audio_data) / sample_rate
        logger.info(f"Audio duration: {total_duration:.2f}s")

        # Process window by window
        position: int = 0
        while position + vad_window_size <= len(audio_data):
            window: np.ndarray = audio_data[position : position + vad_window_size]
            position += vad_window_size

            state.audio_buffer.append(window)

            # VAD check
            window_bytes: bytes = struct.pack(f"{len(window)}h", *window)
            try:
                is_speech: bool = vad.is_speech(window_bytes, sample_rate)
                if is_speech:
                    state.silence_window_count = 0
                    state.speech_window_count += 1
                else:
                    state.silence_window_count += 1

                if VadProcessor.is_segment_complete(
                    state,
                    sample_rate,
                    min_audio_length,
                    min_speech_window_count,
                    max_silence_window_count,
                ):
                    VadProcessor.process_segment(state, sample_rate)

            except Exception:
                logger.exception("VAD error.")

        # Process remaining audio
        if (
            state.audio_buffer
            and sum(chunk.size for chunk in state.audio_buffer) > min_audio_length
        ):
            logger.info("Processing final audio segment...")
            VadProcessor.process_segment(state, sample_rate)

        logger.info(f"Processing complete. Total segments: {state.segment_count}")


def main() -> None:
    model_load_start_time = time.perf_counter()
    vad = webrtcvad.Vad(1)  # aggressiveness level 1
    model_load_duration = time.perf_counter() - model_load_start_time
    logger.info(f"Model load time: {model_load_duration:.3f}s")

    vad_start_time = time.perf_counter()
    VadProcessor.process_audio_file(AUDIO_FILE_PATH, vad)
    vad_duration = time.perf_counter() - vad_start_time
    logger.info(f"VAD processing time: {vad_duration:.3f}s")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()

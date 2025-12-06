import logging
import threading
import time

import numpy as np
import sounddevice as sd
import torch
import whisperx

logger = logging.getLogger(__name__)

SAMPLE_RATE: int = 16000


class WhisperXProcessor:
    def __init__(
        self,
        model_name: str,
        sample_rate: int = SAMPLE_RATE,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        compute_type: str = "float16" if torch.cuda.is_available() else "float32",
    ) -> None:
        self.sample_rate = sample_rate
        self.device = device
        self.compute_type = compute_type

        logger.info(f"Loading WhisperX model '{model_name}' on {device}...")

        self.model = whisperx.load_model(
            model_name,
            device=device,
            compute_type=compute_type,
        )

        # Load alignment model (for word-level timestamps)
        self.align_model, self.align_metadata = whisperx.load_align_model(
            language_code="en",
            device=device,
        )

        # Audio buffers
        self.audio_buffer: list[np.float32] = []
        self.sentence_count: int = 0

        # Simpler buffering approach - let WhisperX handle VAD
        self.buffer_duration: float = 2.0  # seconds
        self.overlap_duration: float = 0.5  # seconds for continuity
        # prevent infinite buffering
        self.max_buffer_duration: float = 10.0

        self.last_process_time: float = time.time()
        # RMS threshold for basic silence detection
        self.silence_threshold: float = 0.01
        # minimum seconds to process
        self.min_audio_length: float = 0.5

    def audio_callback(
        self,
        input_data: np.ndarray,
        frames: int,  # noqa: ARG002
        time_info: dict,  # noqa: ARG002
        status: sd.CallbackFlags | None,
    ) -> None:
        if status:
            logger.warning(f"{status = }")

        # Convert to mono float32
        chunk = input_data[:, 0] if len(input_data.shape) > 1 else input_data.flatten()
        chunk_float32 = chunk.astype(np.float32)

        # Add to buffer
        self.audio_buffer.extend(chunk_float32)

        # Check if we should process
        current_time = time.time()
        buffer_duration = len(self.audio_buffer) / self.sample_rate
        time_since_last = current_time - self.last_process_time

        # Process if:
        # 1. Buffer is getting full, OR
        # 2. We have enough audio and recent silence
        should_process = buffer_duration >= self.max_buffer_duration or (
            buffer_duration >= self.buffer_duration
            and time_since_last >= self.buffer_duration
            and self._has_recent_silence()
        )

        if should_process and buffer_duration >= self.min_audio_length:
            self._process_audio()
            self.last_process_time = current_time

    # Simple silence detection using RMS of recent audio
    def _has_recent_silence(self) -> bool:
        if len(self.audio_buffer) < int(0.5 * self.sample_rate):
            return False

        # Check RMS of last 0.5 seconds
        recent_audio = np.array(self.audio_buffer[-int(0.5 * self.sample_rate) :])
        rms = np.sqrt(np.mean(recent_audio**2))
        return rms < self.silence_threshold

    def _process_audio(self) -> None:
        if not self.audio_buffer:
            return

        # Keep overlap for continuity
        overlap_samples = int(self.overlap_duration * self.sample_rate)
        audio_to_process = np.array(self.audio_buffer, dtype=np.float32)

        # Keep overlap in buffer
        if len(self.audio_buffer) > overlap_samples:
            self.audio_buffer = self.audio_buffer[-overlap_samples:]
        else:
            self.audio_buffer = []

        self.sentence_count += 1

        logger.info(f"🎯 Processing audio segment {self.sentence_count}")

        # Process in separate thread
        thread = threading.Thread(
            target=self._transcribe_with_whisperx,
            args=(audio_to_process, self.sentence_count),
            daemon=True,
        )
        thread.start()

    def _transcribe_with_whisperx(
        self,
        audio_data: np.ndarray,
        segment_num: int,
    ) -> None:
        try:
            duration = len(audio_data) / self.sample_rate
            logger.info(f"Processing segment {segment_num} ({duration:.2f}s)...")

            # WhisperX expects audio as numpy array
            result = self.model.transcribe(
                audio_data,
                batch_size=16,
                language="en",
            )

            # Align for word-level timestamps
            aligned_result = whisperx.align(
                result["segments"],
                self.align_model,
                self.align_metadata,
                audio_data,
                self.device,
                return_char_alignments=False,
            )

            # Extract text and log results
            full_text = " ".join(
                [segment["text"].strip() for segment in aligned_result["segments"]],
            )

            if full_text.strip():
                logger.info(f"🎤 Segment {segment_num}: {full_text}")

                # Optionally log word-level timestamps
                for segment in aligned_result["segments"]:
                    if "words" in segment:
                        words_info = []
                        for word in segment["words"]:
                            start = word.get("start", "?")
                            end = word.get("end", "?")
                            words_info.append(f"{word['word']}({start:.1f}-{end:.1f})")
                        logger.info(f"Words: {' '.join(words_info)}")
            else:
                logger.info(f"🔇 Segment {segment_num}: [No speech detected]")

        except Exception:
            logger.exception(f"Error processing segment {segment_num}")

    def start_recording(self) -> None:
        chunk_size = int(self.sample_rate * 0.1)  # 100ms chunks
        logger.info(
            "Starting recording with WhisperX. Speak naturally, press Ctrl+C to stop.",
        )
        logger.info("Parameters:")
        logger.info(f"  - Sample rate: {self.sample_rate} Hz")
        logger.info(f"  - Buffer duration: {self.buffer_duration}s")
        logger.info(f"  - Overlap duration: {self.overlap_duration}s")
        logger.info(f"  - Device: {self.device}")
        logger.info(f"  - Compute type: {self.compute_type}")

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
            if len(self.audio_buffer) > int(self.min_audio_length * self.sample_rate):
                logger.info("Processing final audio segment...")
                self._process_audio()


def main() -> None:
    processor = WhisperXProcessor(
        model_name="medium.en",  # or "large-v2", "small.en", etc.
    )
    processor.start_recording()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()

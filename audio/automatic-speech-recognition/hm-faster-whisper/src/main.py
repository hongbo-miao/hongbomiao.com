import logging
from pathlib import Path

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


def main(audio_path: Path, transcription_file: Path) -> None:
    model = WhisperModel(
        model_size_or_path="medium",
        device="cpu",
        compute_type="float32",
    )

    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 400},
        language="en",
        task="transcribe",
        initial_prompt=None,
        word_timestamps=False,
    )

    logger.info(f"{info = }")

    transcription = ""
    for segment in segments:
        logger.info(f"[{segment.start:.1f}s -> {segment.end:.1f}s] {segment.text}")
        transcription += segment.text + " "

    transcription_file.write_text(transcription.strip(), encoding="utf-8")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    data_dir_path = Path("data")
    audio_path = data_dir_path / Path("audio.mp3")

    output_dir_path = Path("output")
    output_dir_path.mkdir(exist_ok=True)
    transcription_path = output_dir_path / Path("transcription.txt")

    main(audio_path, transcription_path)

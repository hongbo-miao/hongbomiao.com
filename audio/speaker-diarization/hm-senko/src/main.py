import logging
from pathlib import Path

import senko

logger = logging.getLogger(__name__)


def main(wav_path: Path) -> None:
    diarizer = senko.Diarizer(device="auto", warmup=False, quiet=False)
    result = diarizer.diarize(str(wav_path), generate_colors=False)
    logger.info(result)
    senko.save_json(result["merged_segments"], "data/audio_diarized.json")
    senko.save_rttm(
        result["merged_segments"],
        wav_path,
        "data/audio_diarized.rttm",
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    # Must be a 16 kHz, mono, 16-bit WAV file
    wav_path = Path("/path/to/audio.wav")
    main(wav_path)

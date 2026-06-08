import logging
import sys
from pathlib import Path

from audio.utils.score_wav_file import score_wav_file
from file.utils.collect_wav_paths import collect_wav_paths
from report.utils.format_comparison import format_comparison

logger = logging.getLogger(__name__)

INPUT_DIRECTORY = Path("data")


def main() -> int:
    wav_paths = collect_wav_paths(INPUT_DIRECTORY)
    if len(wav_paths) < 2:
        logger.error(f"Need at least two WAV files to compare in {INPUT_DIRECTORY}")
        return 1

    results = []
    for wav_path in wav_paths:
        try:
            results.append(score_wav_file(wav_path))
        except Exception:
            logger.exception(f"Failed to score {wav_path}")

    if len(results) < 2:
        logger.error("Need at least two successfully scored WAV files to compare")
        return 1

    logger.info(format_comparison(results))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(main())

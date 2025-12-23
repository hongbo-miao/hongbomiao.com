import logging

from shared.utils.parse_segment import parse_segment

logger = logging.getLogger(__name__)


def log_diarization_segments(segments: list[str]) -> None:
    if not segments:
        logger.warning("Diarization produced no segments.")
        return

    parsed_segments: list[tuple[float, float, str]] = [
        parse_segment(segment) for segment in segments
    ]
    for start_time, end_time, speaker_identifier in sorted(parsed_segments):
        logger.info(
            f"Speaker {speaker_identifier} from {start_time:.2f}s to {end_time:.2f}s "
            f"(duration {end_time - start_time:.2f}s)",
        )

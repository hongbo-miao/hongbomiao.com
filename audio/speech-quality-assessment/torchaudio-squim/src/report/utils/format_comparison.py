from audio.types.audio_quality_result import AudioQualityResult
from report.utils.format_result_row import format_result_row

HEADER = (
    f"{'file':<12} {'dur (s)':>8} {'SNR (dB)':>9} "
    f"{'speech %':>9} {'est PESQ':>9} {'est STOI':>9} {'est SI-SDR':>11}"
)


def format_comparison(results: list[AudioQualityResult]) -> str:
    separator = "-" * len(HEADER)
    rows = [format_result_row(result) for result in results]
    return "\n".join([HEADER, separator, *rows])

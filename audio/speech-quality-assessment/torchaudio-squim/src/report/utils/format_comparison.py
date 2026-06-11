from audio.types.audio_quality_result import AudioQualityResult
from report.utils.format_result_row import format_result_row


def format_comparison(results: list[AudioQualityResult]) -> str:
    name_column_width = max([len("file")] + [len(result.name) for result in results])
    header = (
        f"{'file':<{name_column_width}} {'dur (s)':>8} {'SNR (dB)':>9} "
        f"{'speech %':>9} {'est PESQ':>9} {'est STOI':>9} {'est SI-SDR':>11}"
    )
    separator = "-" * len(header)
    rows = [format_result_row(result, name_column_width) for result in results]
    return "\n".join([header, separator, *rows])

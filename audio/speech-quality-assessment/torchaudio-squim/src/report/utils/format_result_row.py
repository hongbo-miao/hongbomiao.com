from audio.types.audio_quality_result import AudioQualityResult


def format_result_row(result: AudioQualityResult) -> str:
    pesq_text = (
        f"{result.estimated_pesq:.2f}" if result.estimated_pesq is not None else "n/a"
    )
    stoi_text = (
        f"{result.estimated_stoi:.2f}" if result.estimated_stoi is not None else "n/a"
    )
    si_sdr_text = (
        f"{result.estimated_si_sdr_db:.1f}"
        if result.estimated_si_sdr_db is not None
        else "n/a"
    )
    return (
        f"{result.name:<12.12} {result.duration_seconds:>8.1f} "
        f"{result.silence_gap_snr_db:>9.1f} {result.speech_fraction * 100:>8.1f}% "
        f"{pesq_text:>9} {stoi_text:>9} {si_sdr_text:>11}"
    )

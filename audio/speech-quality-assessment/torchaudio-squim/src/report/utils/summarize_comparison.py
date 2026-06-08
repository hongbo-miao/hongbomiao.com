from audio.types.audio_quality_result import AudioQualityResult

# Below this SNR difference the recordings count as comparable rather than one being clearly better.
SNR_TIE_THRESHOLD_DB = 0.5


def summarize_comparison(results: list[AudioQualityResult]) -> str:
    if len(results) == 0:
        return "No recordings to compare."
    if len(results) == 1:
        return f"{results[0].name} is the only recording; no comparison available."

    best = max(results, key=lambda result: result.silence_gap_snr_db)
    worst = min(results, key=lambda result: result.silence_gap_snr_db)
    spread = best.silence_gap_snr_db - worst.silence_gap_snr_db

    if spread < SNR_TIE_THRESHOLD_DB:
        return f"All recordings have a comparable silence-gap SNR (within {spread:.1f} dB)."
    if len(results) == 2:
        return f"{best.name} has the higher silence-gap SNR (by {spread:.1f} dB)."
    return (
        f"{best.name} has the highest silence-gap SNR ({best.silence_gap_snr_db:.1f} dB) "
        f"and {worst.name} the lowest ({worst.silence_gap_snr_db:.1f} dB)."
    )

from pathlib import Path

from audio.types.audio_quality_result import AudioQualityResult
from audio.utils.compute_silence_gap_snr_db import compute_silence_gap_snr_db
from audio.utils.compute_squim_metrics import compute_squim_metrics
from audio.utils.load_mono_audio import load_mono_audio


def score_wav_file(wav_path: Path) -> AudioQualityResult:
    samples, sample_rate_hz = load_mono_audio(wav_path)
    if sample_rate_hz <= 0:
        message = f"Invalid sample rate {sample_rate_hz} for {wav_path}"
        raise ValueError(message)
    duration_seconds = len(samples) / sample_rate_hz
    snr_db, speech_fraction = compute_silence_gap_snr_db(samples, sample_rate_hz)

    squim = compute_squim_metrics(samples, sample_rate_hz)
    estimated_pesq = squim[0] if squim else None
    estimated_stoi = squim[1] if squim else None
    estimated_si_sdr_db = squim[2] if squim else None

    return AudioQualityResult(
        name=wav_path.stem,
        duration_seconds=duration_seconds,
        silence_gap_snr_db=snr_db,
        speech_fraction=speech_fraction,
        estimated_pesq=estimated_pesq,
        estimated_stoi=estimated_stoi,
        estimated_si_sdr_db=estimated_si_sdr_db,
    )

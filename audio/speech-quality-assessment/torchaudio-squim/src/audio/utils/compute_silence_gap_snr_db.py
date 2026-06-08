import numpy as np
from audio.constants.scoring_constants import QUIET_PERCENTILE, SPEECH_PERCENTILE
from audio.utils.compute_frame_powers import compute_frame_powers


def compute_silence_gap_snr_db(
    samples: np.ndarray,
    sample_rate_hz: int,
) -> tuple[float, float]:
    """
    Estimate SNR by comparing loud (speech) frames against quiet (noise) frames.

    Returns the SNR in dB and the fraction of frames classified as speech.
    It is reference-free and deterministic, relying on the natural pauses in speech.
    """
    powers = compute_frame_powers(samples, sample_rate_hz)
    epsilon = 1e-12
    quiet_threshold = np.percentile(powers, QUIET_PERCENTILE)
    speech_threshold = np.percentile(powers, SPEECH_PERCENTILE)

    noise_powers = powers[powers <= quiet_threshold]
    speech_powers = powers[powers >= speech_threshold]
    noise_power = float(np.mean(noise_powers)) if noise_powers.size else epsilon
    speech_power = float(np.mean(speech_powers)) if speech_powers.size else epsilon

    snr_db = 10.0 * np.log10((speech_power + epsilon) / (noise_power + epsilon))
    speech_fraction = float(np.mean(powers >= speech_threshold))
    return snr_db, speech_fraction

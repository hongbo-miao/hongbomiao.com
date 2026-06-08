import functools
import logging

import numpy as np
from audio.constants.scoring_constants import (
    SQUIM_SAMPLE_RATE_HZ,
    SQUIM_WINDOW_SECONDS,
)

try:
    import torch
    import torchaudio
    from torchaudio.pipelines import SQUIM_OBJECTIVE
except ImportError:
    torch = None
    torchaudio = None
    SQUIM_OBJECTIVE = None

logger = logging.getLogger(__name__)

# A window must hold at least half a second of speech to yield a meaningful SQUIM score.
MINIMUM_WINDOW_SECONDS = 0.5

# Floor for the speech-power threshold so an all-zero or near-silent recording
# does not collapse the threshold to zero and let silent windows reach SQUIM.
MINIMUM_SPEECH_POWER = 1e-10


@functools.cache
def load_squim_objective_model() -> "torchaudio.models.SquimObjective":
    # Cache the model so scoring many recordings loads the checkpoint only once.
    return SQUIM_OBJECTIVE.get_model()


def compute_squim_metrics(
    samples: np.ndarray,
    sample_rate_hz: int,
) -> tuple[float, float, float] | None:
    """
    Predict PESQ, STOI, and SI-SDR without a reference using TorchAudio SQUIM.

    Returns None if TorchAudio or the model weights are unavailable, so the caller can fall back to the silence-gap SNR alone.
    SQUIM was trained on wideband speech, so for band-limited (for example telephone-band) audio the absolute values read low.
    They are most useful as a relative comparison between recordings.
    """
    if torch is None or torchaudio is None or SQUIM_OBJECTIVE is None:
        return None

    try:
        waveform = torch.from_numpy(samples).float().unsqueeze(0)
        if sample_rate_hz != SQUIM_SAMPLE_RATE_HZ:
            waveform = torchaudio.functional.resample(
                waveform,
                sample_rate_hz,
                SQUIM_SAMPLE_RATE_HZ,
            )
        model = load_squim_objective_model()
        window_length = int(SQUIM_WINDOW_SECONDS * SQUIM_SAMPLE_RATE_HZ)
        minimum_window_length = int(MINIMUM_WINDOW_SECONDS * SQUIM_SAMPLE_RATE_HZ)
        mean_power = float(torch.mean(waveform**2))
        speech_threshold = max(mean_power * 0.5, MINIMUM_SPEECH_POWER)
        pesq_scores: list[float] = []
        stoi_scores: list[float] = []
        si_sdr_scores: list[float] = []
        if waveform.shape[1] < window_length:
            windows = [waveform]
        else:
            windows = [
                waveform[:, index * window_length : (index + 1) * window_length]
                for index in range(waveform.shape[1] // window_length)
            ]
        with torch.no_grad():
            for window in windows:
                too_short = window.shape[1] < minimum_window_length
                too_quiet = float(torch.mean(window**2)) < speech_threshold
                if too_short or too_quiet:
                    continue
                stoi, pesq, si_sdr = model(window)
                pesq_scores.append(float(pesq.item()))
                stoi_scores.append(float(stoi.item()))
                si_sdr_scores.append(float(si_sdr.item()))
        if not pesq_scores:
            return None
        return (
            float(np.mean(pesq_scores)),
            float(np.mean(stoi_scores)),
            float(np.mean(si_sdr_scores)),
        )
    except Exception:
        logger.exception("SQUIM scoring failed; reporting silence-gap SNR only")
        return None

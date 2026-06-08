import numpy as np
from audio.constants.scoring_constants import FRAME_DURATION_SECONDS


def compute_frame_powers(samples: np.ndarray, sample_rate_hz: int) -> np.ndarray:
    frame_length = max(1, int(FRAME_DURATION_SECONDS * sample_rate_hz))
    frame_count = len(samples) // frame_length
    if frame_count == 0:
        return (
            np.array([float(np.mean(samples**2))]) if len(samples) else np.array([0.0])
        )
    trimmed = samples[: frame_count * frame_length].reshape(frame_count, frame_length)
    return np.mean(trimmed**2, axis=1)

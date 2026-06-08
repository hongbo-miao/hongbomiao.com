from pathlib import Path

import numpy as np
import soundfile


def load_mono_audio(wav_path: Path) -> tuple[np.ndarray, int]:
    samples, sample_rate_hz = soundfile.read(str(wav_path), dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return samples, sample_rate_hz

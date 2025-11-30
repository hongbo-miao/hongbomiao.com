from pathlib import Path

import librosa
import numpy as np


def load_audio_file(audio_file_path: Path) -> tuple[np.ndarray, int]:
    if not audio_file_path.exists():
        msg = f"Audio file not found: {audio_file_path}"
        raise FileNotFoundError(msg)

    audio_data, sample_rate = librosa.load(audio_file_path)
    return audio_data, sample_rate

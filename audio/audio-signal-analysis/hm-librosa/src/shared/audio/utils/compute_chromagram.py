import librosa
import numpy as np


def compute_chromagram(audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
    return librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)

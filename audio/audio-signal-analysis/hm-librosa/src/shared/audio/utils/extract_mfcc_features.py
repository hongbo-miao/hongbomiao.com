import librosa
import numpy as np


def extract_mfcc_features(audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
    return librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)

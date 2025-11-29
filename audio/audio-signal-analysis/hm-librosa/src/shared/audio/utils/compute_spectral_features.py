import librosa
import numpy as np


def compute_spectral_features(
    audio_data: np.ndarray,
    sample_rate: int,
) -> dict[str, np.ndarray]:
    spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio_data,
        sr=sample_rate,
    )
    spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sample_rate)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)
    root_mean_square = librosa.feature.rms(y=audio_data)

    return {
        "spectral_centroid": spectral_centroid,
        "spectral_rolloff": spectral_rolloff,
        "spectral_bandwidth": spectral_bandwidth,
        "spectral_contrast": spectral_contrast,
        "zero_crossing_rate": zero_crossing_rate,
        "root_mean_square": root_mean_square,
    }

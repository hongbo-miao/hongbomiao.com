import librosa
import numpy as np


def compute_mel_spectrogram(audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
    mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
    return librosa.power_to_db(mel_spectrogram, ref=np.max)

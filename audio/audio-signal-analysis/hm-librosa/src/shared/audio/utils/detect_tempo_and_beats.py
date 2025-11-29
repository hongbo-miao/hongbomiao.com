import librosa
import numpy as np


def detect_tempo_and_beats(
    audio_data: np.ndarray,
    sample_rate: int,
) -> tuple[float, np.ndarray]:
    tempo, beat_frames = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
    tempo_value = float(np.squeeze(tempo))
    beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate)
    return tempo_value, beat_times

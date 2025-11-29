import logging
from pathlib import Path

import numpy as np
from shared.audio.utils.compute_chromagram import compute_chromagram
from shared.audio.utils.compute_mel_spectrogram import compute_mel_spectrogram
from shared.audio.utils.compute_spectral_features import compute_spectral_features
from shared.audio.utils.detect_tempo_and_beats import detect_tempo_and_beats
from shared.audio.utils.extract_mfcc_features import extract_mfcc_features
from shared.audio.utils.load_audio_file import load_audio_file

logger = logging.getLogger(__name__)
AUDIO_FILE_PATH = Path(__file__).resolve().parent.parent / "data" / "audio.wav"


def main() -> None:
    audio_data, sample_rate = load_audio_file(AUDIO_FILE_PATH)
    logger.info(f"Loaded {AUDIO_FILE_PATH.name} with sample rate {sample_rate} Hz")
    logger.info(f"Audio duration: {len(audio_data) / sample_rate} seconds")

    logger.info("# MFCC features")
    mfcc_features = extract_mfcc_features(audio_data, sample_rate)
    logger.info(f"MFCC feature matrix shape: {mfcc_features.shape}")
    logger.info(f"First five MFCC mean values: {np.mean(mfcc_features, axis=1)[:5]}")

    logger.info("# Tempo and beats")
    tempo, beat_times = detect_tempo_and_beats(audio_data, sample_rate)
    logger.info(f"Estimated tempo: {tempo} BPM")
    logger.info(f"Detected beat count: {len(beat_times)}")

    logger.info("# Spectral features")
    spectral_features = compute_spectral_features(audio_data, sample_rate)
    logger.info(
        f"Spectral centroid mean: {np.mean(spectral_features['spectral_centroid'])} Hz",
    )
    logger.info(
        f"Spectral rolloff mean: {np.mean(spectral_features['spectral_rolloff'])} Hz",
    )
    logger.info(
        "Spectral bandwidth mean: "
        f"{np.mean(spectral_features['spectral_bandwidth'])} Hz",
    )
    logger.info(
        "Spectral contrast mean: "
        f"{np.mean(spectral_features['spectral_contrast']):.4f}",
    )
    logger.info(
        "Zero crossing rate mean: "
        f"{np.mean(spectral_features['zero_crossing_rate']):.4f}",
    )
    logger.info(
        "Root mean square energy mean: "
        f"{np.mean(spectral_features['root_mean_square']):.4f}",
    )

    logger.info("# Chromagram")
    chromagram = compute_chromagram(audio_data, sample_rate)
    logger.info(f"Chromagram shape: {chromagram.shape}")

    logger.info("# Mel spectrogram")
    mel_spectrogram = compute_mel_spectrogram(audio_data, sample_rate)
    logger.info(
        "Mel spectrogram range: ["
        f"{np.min(mel_spectrogram)}, {np.max(mel_spectrogram)}] dB",
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()

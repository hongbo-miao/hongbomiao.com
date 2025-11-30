import logging
from pathlib import Path

import numpy as np
from shared.audio.utils.compute_chromagram import compute_chromagram
from shared.audio.utils.compute_mel_spectrogram import compute_mel_spectrogram
from shared.audio.utils.compute_spectral_features import compute_spectral_features
from shared.audio.utils.detect_tempo_and_beats import detect_tempo_and_beats
from shared.audio.utils.extract_mfcc_features import extract_mfcc_features
from shared.file.utils.load_audio_file import load_audio_file
from shared.visualization.utils.visualize_chromagram import visualize_chromagram
from shared.visualization.utils.visualize_mel_spectrogram import (
    visualize_mel_spectrogram,
)
from shared.visualization.utils.visualize_mfcc import visualize_mfcc
from shared.visualization.utils.visualize_waveform import visualize_waveform

logger = logging.getLogger(__name__)

AUDIO_FILE_PATH = Path("data/audio.wav")
OUTPUT_DIRECTORY_PATH = Path("output")


def main() -> None:
    audio_data, sample_rate = load_audio_file(AUDIO_FILE_PATH)
    logger.info(f"Loaded {AUDIO_FILE_PATH.name} with sample rate {sample_rate} Hz")
    logger.info(f"Audio duration: {len(audio_data) / sample_rate} seconds")

    logger.info("# MFCC features")
    mfcc_features = extract_mfcc_features(audio_data, sample_rate)
    logger.info(f"MFCC feature matrix shape: {mfcc_features.shape}")
    logger.info(f"First five MFCC mean values: {np.mean(mfcc_features, axis=1)[:5]}")
    visualize_mfcc(mfcc_features, OUTPUT_DIRECTORY_PATH / "mfcc.png")

    logger.info("# Tempo and beats")
    tempo, beat_times = detect_tempo_and_beats(audio_data, sample_rate)
    logger.info(f"Estimated tempo: {tempo} BPM")
    logger.info(f"Detected beat count: {len(beat_times)}")
    visualize_waveform(
        audio_data,
        sample_rate,
        OUTPUT_DIRECTORY_PATH / "waveform.png",
        beat_times,
    )

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
    visualize_chromagram(chromagram, OUTPUT_DIRECTORY_PATH / "chromagram.png")

    logger.info("# Mel spectrogram")
    mel_spectrogram = compute_mel_spectrogram(audio_data, sample_rate)
    logger.info(
        "Mel spectrogram range: ["
        f"{np.min(mel_spectrogram)}, {np.max(mel_spectrogram)}] dB",
    )
    visualize_mel_spectrogram(
        mel_spectrogram,
        OUTPUT_DIRECTORY_PATH / "mel_spectrogram.png",
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()

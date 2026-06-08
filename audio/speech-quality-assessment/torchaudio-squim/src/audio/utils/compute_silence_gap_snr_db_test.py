import numpy as np
from audio.utils.compute_silence_gap_snr_db import compute_silence_gap_snr_db


def make_bursty_speech(
    noise_amplitude: float,
    sample_rate_hz: int = 48_000,
) -> np.ndarray:
    """Build alternating speech-burst and silence blocks over a constant noise floor."""
    generator = np.random.default_rng(0)
    blocks = []
    for index in range(12):
        block_length = sample_rate_hz // 2
        noise = generator.normal(0, noise_amplitude, block_length).astype(np.float32)
        if index % 2 == 0:
            time = np.arange(block_length) / sample_rate_hz
            speech = 0.3 * (
                np.sin(2 * np.pi * 500 * time) + np.sin(2 * np.pi * 1500 * time)
            )
            blocks.append(speech.astype(np.float32) / 2 + noise)
        else:
            blocks.append(noise)
    return np.clip(np.concatenate(blocks), -1.0, 1.0)


class TestComputeSilenceGapSnrDb:
    def test_clean_signal_scores_higher_than_noisy(self) -> None:
        sample_rate_hz = 48_000
        clean_snr_db, _ = compute_silence_gap_snr_db(
            make_bursty_speech(0.01, sample_rate_hz),
            sample_rate_hz,
        )
        noisy_snr_db, _ = compute_silence_gap_snr_db(
            make_bursty_speech(0.1, sample_rate_hz),
            sample_rate_hz,
        )
        assert clean_snr_db > noisy_snr_db
        assert clean_snr_db - noisy_snr_db > 10.0

    def test_speech_fraction_is_a_ratio(self) -> None:
        sample_rate_hz = 48_000
        _, speech_fraction = compute_silence_gap_snr_db(
            make_bursty_speech(0.01, sample_rate_hz),
            sample_rate_hz,
        )
        assert 0.0 <= speech_fraction <= 1.0

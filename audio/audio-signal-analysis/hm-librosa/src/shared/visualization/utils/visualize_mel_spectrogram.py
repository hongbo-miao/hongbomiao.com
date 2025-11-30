from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def visualize_mel_spectrogram(
    mel_spectrogram: np.ndarray,
    output_file_path: Path,
) -> None:
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots()
    image = axis.imshow(mel_spectrogram, aspect="auto", origin="lower")
    axis.set_xlabel("Frame")
    axis.set_ylabel("Mel bin")
    axis.set_title("Mel spectrogram (dB)")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output_file_path, dpi=150)
    plt.close(figure)

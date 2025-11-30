from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PITCH_CLASS_LABELS = (
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
)


def visualize_chromagram(
    chromagram: np.ndarray,
    output_file_path: Path,
) -> None:
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots()
    image = axis.imshow(chromagram, aspect="auto", origin="lower")
    axis.set_xlabel("Frame")
    axis.set_ylabel("Pitch class")
    axis.set_title("Chromagram")
    axis.set_yticks(np.arange(len(PITCH_CLASS_LABELS)))
    axis.set_yticklabels(PITCH_CLASS_LABELS)
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output_file_path, dpi=150)
    plt.close(figure)

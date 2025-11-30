from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def visualize_waveform(
    audio_data: np.ndarray,
    sample_rate: int,
    output_file_path: Path,
    beat_times: np.ndarray | None = None,
) -> None:
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    time_data = np.arange(len(audio_data)) / sample_rate

    figure, axis = plt.subplots()
    axis.plot(time_data, audio_data, label="Waveform")

    if beat_times is not None and beat_times.size > 0:
        axis.vlines(
            x=beat_times,
            ymin=audio_data.min(),
            ymax=audio_data.max(),
            color="red",
            alpha=0.3,
        )

    axis.set_xlabel("Time (seconds)")
    axis.set_ylabel("Amplitude")
    axis.set_title("Waveform")
    axis.grid(visible=True)
    figure.tight_layout()
    figure.savefig(output_file_path, dpi=150)
    plt.close(figure)

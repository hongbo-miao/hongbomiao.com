import logging
from pathlib import Path

import matplotlib.pyplot as plt
import torch

logger = logging.getLogger(__name__)


def visualize_results(
    target_data: torch.Tensor,
    generated_samples: torch.Tensor,
) -> None:
    """Plot target data and generated samples side by side."""
    _figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(target_data[:, 0], target_data[:, 1], alpha=0.5, s=10)
    axes[0].set_title("Target Data (Two Moons)")
    axes[0].set_xlim(-2, 3)
    axes[0].set_ylim(-1.5, 2)
    axes[0].set_aspect("equal")

    axes[1].scatter(generated_samples[:, 0], generated_samples[:, 1], alpha=0.5, s=10)
    axes[1].set_title("Generated Samples (Flow Matching)")
    axes[1].set_xlim(-2, 3)
    axes[1].set_ylim(-1.5, 2)
    axes[1].set_aspect("equal")

    output_directory_path = Path("output")
    output_file_path = output_directory_path / "flow_matching_results.png"

    plt.tight_layout()
    plt.savefig(output_file_path, dpi=150)
    logger.info(f"Saved visualization to {output_file_path}")
    plt.show()

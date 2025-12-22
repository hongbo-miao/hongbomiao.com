import logging
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from shared.services.variational_autoencoder import VAE

logger = logging.getLogger(__name__)


def generate_samples(
    model: VAE,
    device: torch.device,
    latent_dim: int,
    output_path: Path,
    num_samples: int = 16,
) -> Path:
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        samples = model.decode(z)

    grid_size = int(num_samples**0.5)
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
    axes = axes.flat if num_samples > 1 else [axes]

    for i, ax in enumerate(axes):
        if i >= num_samples:
            break
        ax.imshow(samples[i].cpu().view(28, 28), cmap="gray")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    logger.info("Generated samples saved to %s", output_path)
    return output_path

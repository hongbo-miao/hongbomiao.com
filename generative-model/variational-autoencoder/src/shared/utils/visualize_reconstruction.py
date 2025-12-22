import logging
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from shared.services.variational_autoencoder import VAE
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def visualize_reconstruction(
    model: VAE,
    device: torch.device,
    test_loader: DataLoader,
    output_path: Path,
    image_count: int = 8,
) -> Path:
    model.eval()
    with torch.no_grad():
        data, _ = next(iter(test_loader))
        data = data.to(device)
        recon, _, _ = model(data)

    n = min(data.size(0), image_count)
    fig, axes = plt.subplots(2, n, figsize=(15, 3))

    for i in range(n):
        axes[0, i].imshow(data[i].cpu().squeeze(), cmap="gray")
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Original", fontsize=10)

        axes[1, i].imshow(recon.view(-1, 1, 28, 28)[i].cpu().squeeze(), cmap="gray")
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("Reconstructed", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    logger.info(f"Reconstruction saved to {output_path}")
    return output_path

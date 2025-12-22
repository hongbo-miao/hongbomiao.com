import logging
from pathlib import Path

import torch
from shared.services.train_variational_autoencoder import train_variational_autoencoder
from shared.services.variational_autoencoder import VAE
from shared.utils.generate_samples import generate_samples
from shared.utils.get_device import get_device
from shared.utils.visualize_reconstruction import visualize_reconstruction
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

BATCH_SIZE = 128
EPOCH_COUNT = 10
LATENT_DIMENSION = 20
LEARNING_RATE = 1e-3
RECONSTRUCTION_PATH = OUTPUT_DIR / "vae_reconstruction.png"
GENERATED_PATH = OUTPUT_DIR / "vae_generated.png"


def prepare_dataloaders() -> tuple[DataLoader, DataLoader]:
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(
        DATA_DIR,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(DATA_DIR, train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    logger.info("Using device: %s", device)

    train_loader, test_loader = prepare_dataloaders()

    model = VAE(latent_dim=LATENT_DIMENSION).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    logger.info("Starting training...")
    train_variational_autoencoder(
        model=model,
        device=device,
        train_loader=train_loader,
        optimizer=optimizer,
        epoch_count=EPOCH_COUNT,
    )

    logger.info("Generating visualizations...")
    visualize_reconstruction(
        model=model,
        device=device,
        test_loader=test_loader,
        output_path=RECONSTRUCTION_PATH,
    )
    generate_samples(
        model=model,
        device=device,
        latent_dim=LATENT_DIMENSION,
        output_path=GENERATED_PATH,
        num_samples=16,
    )
    logger.info("Training complete!")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()

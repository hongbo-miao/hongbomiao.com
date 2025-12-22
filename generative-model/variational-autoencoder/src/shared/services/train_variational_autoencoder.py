import torch
from shared.services.train_epoch import train_epoch
from shared.services.variational_autoencoder import VAE
from torch.utils.data import DataLoader


def train_variational_autoencoder(
    model: VAE,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch_count: int,
) -> list[float]:
    epoch_losses: list[float] = []
    for epoch in range(1, epoch_count + 1):
        avg_loss = train_epoch(
            model=model,
            device=device,
            train_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
        )
        epoch_losses.append(avg_loss)
    return epoch_losses

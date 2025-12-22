import logging

import torch
from shared.services.loss_function import loss_function
from shared.services.variational_autoencoder import VAE
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def train_epoch(
    model: VAE,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
) -> float:
    model.train()
    train_loss = 0.0

    for batch_idx, (batch, _) in enumerate(train_loader):
        data = batch.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if batch_idx % 100 == 0:
            logger.info(
                "Epoch %s [%s/%s] Loss: %.4f",
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                loss.item() / len(data),
            )

    avg_loss = train_loss / len(train_loader.dataset)
    logger.info("====> Epoch %s Average loss: %.4f", epoch, avg_loss)
    return avg_loss

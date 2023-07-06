import lightning as L
import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision as tv
from args import get_args
from lightning.pytorch.loggers.wandb import WandbLogger


class LitAutoEncoder(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 28 * 28)
        )

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def main():
    project_name = "classify-fashion-mnist"

    # W&B
    wandb_logger = WandbLogger(project=project_name)

    # MLflow
    mlflow.set_tracking_uri("https://mlflow.hongbomiao.com")
    mlflow.set_experiment(experiment_name=project_name)
    mlflow.pytorch.autolog()

    args = get_args()

    dataset = tv.datasets.MNIST(
        "./data", download=True, transform=tv.transforms.ToTensor()
    )
    train, val = data.random_split(dataset, [55000, 5000])

    autoencoder = LitAutoEncoder()
    trainer = L.Trainer(
        devices="auto",
        accelerator="auto",
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=1,
        logger=wandb_logger,
    )
    trainer.fit(autoencoder, data.DataLoader(train), data.DataLoader(val))


if __name__ == "__main__":
    main()

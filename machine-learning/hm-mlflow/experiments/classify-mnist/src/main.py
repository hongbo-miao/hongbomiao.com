import lightning as L  # noqa: N812
import mlflow
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from args import get_args
from lightning.pytorch.loggers.wandb import WandbLogger
from torch import nn
from torch.utils import data


class LitAutoEncoder(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        _batch_idx: int,
    ) -> torch.Tensor:
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def main() -> None:
    project_name = "classify-mnist"

    # W&B
    wandb_logger = WandbLogger(project=project_name)

    # MLflow
    mlflow.set_tracking_uri("https://mlflow.hongbomiao.com")
    mlflow.set_experiment(experiment_name=project_name)
    mlflow.pytorch.autolog()

    args = get_args()

    dataset = torchvision.datasets.MNIST(
        "data/",
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.transforms.ToTensor(),
                torchvision.transforms.transforms.Normalize((0.1307,), (0.3081,)),
            ],
        ),
    )
    train_dataset, val_dataset = data.random_split(dataset, [55000, 5000])

    autoencoder = LitAutoEncoder()
    trainer = L.Trainer(
        devices="auto",
        accelerator="auto",
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=1,
        logger=wandb_logger,
    )
    trainer.fit(
        autoencoder,
        data.DataLoader(train_dataset),
        data.DataLoader(val_dataset),
    )


if __name__ == "__main__":
    main()

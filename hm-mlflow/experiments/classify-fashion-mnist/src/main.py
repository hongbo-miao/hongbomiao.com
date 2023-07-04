import lightning as L
import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision as tv


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
    mlflow.set_tracking_uri("https://mlflow.hongbomiao.com")
    mlflow.pytorch.autolog()
    dataset = tv.datasets.MNIST(
        "./data", download=True, transform=tv.transforms.ToTensor()
    )
    train, val = data.random_split(dataset, [55000, 5000])

    autoencoder = LitAutoEncoder()
    trainer = L.Trainer()
    trainer.fit(autoencoder, data.DataLoader(train), data.DataLoader(val))


if __name__ == "__main__":
    main()

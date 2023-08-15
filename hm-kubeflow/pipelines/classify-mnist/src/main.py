from kfp import client, dsl


@dsl.component(
    base_image="docker.io/python:3.10",
    packages_to_install=["torch==2.0.0", "torchvision==0.15.1", "lightning==2.0.5"],
)
def train():
    import lightning as L
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.utils.data as data
    import torchvision

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

    dataset = torchvision.datasets.MNIST(
        "data/",
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.transforms.ToTensor(),
                torchvision.transforms.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    )
    train, val = data.random_split(dataset, [55000, 5000])

    autoencoder = LitAutoEncoder()
    trainer = L.Trainer(
        devices="auto",
        accelerator="auto",
        max_epochs=2,
        check_val_every_n_epoch=1,
    )
    trainer.fit(autoencoder, data.DataLoader(train), data.DataLoader(val))


@dsl.pipeline
def classify_fashion_mnist():
    train()


if __name__ == "__main__":
    kfp_client = client.Client(host="https://kubeflow.hongbomiao.com")
    run = kfp_client.create_run_from_pipeline_func(classify_fashion_mnist)

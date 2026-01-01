import logging
from pathlib import Path

import torch
import wandb
import yaml
from evaluate import evaluate
from model.data_loader import train_data_loader, val_data_loader
from model.net import Net
from torch import nn, optim
from train import train
from utils.device import device
from utils.writer import write_params

logger = logging.getLogger(__name__)


def main() -> None:
    with Path("params.yaml").open("r") as f:
        params = yaml.safe_load(f)

    write_params(params)

    with wandb.init(
        entity="hongbo-miao",
        project="convolutional-neural-network",
        config=params,
    ) as wb:
        config = wb.config
        net = Net().to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=config["lr"])

        max_val_acc = 0.0
        for epoch in range(config["train"]["epochs"]):
            train_loss = train(net, train_data_loader, device, optimizer, criterion)
            train_acc = evaluate(net, train_data_loader, device)
            val_acc = evaluate(net, val_data_loader, device)

            logger.info({"Train": train_acc, "Validation": val_acc})
            wb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                },
            )
            if val_acc > max_val_acc:
                logger.info("Found better model.")
                max_val_acc = val_acc

                filename = "output/models/model.pt"
                torch.save(net.state_dict(), filename)
                wb.save(filename)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()

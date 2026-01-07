import logging

import hydra
import orjson
import torch
from config.constants import INPUT_SIZE, OUTPUT_SIZE
from config.register_config import register_config
from hydra_zen import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

register_config()


@hydra.main(config_name="config", version_base="1.3", config_path=None)
def main(config: DictConfig) -> None:
    logger.info(
        f"Config:\n{orjson.dumps(OmegaConf.to_container(config), option=orjson.OPT_INDENT_2).decode()}",
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu",
    )
    logger.info(f"Using device: {device}")

    activation = instantiate(config.activation)

    model = instantiate(config.model, activation=activation)
    model = model.to(device)

    optimizer = instantiate(config.optimizer, params=model.parameters())

    sample_count = 1000
    x_train = torch.randn(sample_count, INPUT_SIZE, device=device)
    y_train = torch.randint(0, OUTPUT_SIZE, (sample_count,), device=device)

    criterion = nn.CrossEntropyLoss()

    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )

    model.train()
    for epoch in range(config.epoch_count):
        total_loss = 0.0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        logger.info(f"Epoch {epoch + 1}/{config.epoch_count}, Loss: {average_loss:.4f}")

    model.eval()
    with torch.no_grad():
        outputs = model(x_train)
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == y_train).float().mean().item()

    logger.info(f"Final accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()

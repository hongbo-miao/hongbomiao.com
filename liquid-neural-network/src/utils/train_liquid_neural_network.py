import logging

import torch
from models.liquid_neural_network import LiquidNeuralNetwork
from torch import nn

logger = logging.getLogger(__name__)


def train_liquid_neural_network(
    model: LiquidNeuralNetwork,
    input_data: torch.Tensor,
    target_data: torch.Tensor,
    epoch_count: int,
    learning_rate: float,
) -> None:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epoch_count):
        model.train()
        optimizer.zero_grad()
        prediction = model(input_data)
        loss = criterion(prediction, target_data)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            logger.info(f"Epoch {epoch + 1}/{epoch_count}, Loss: {loss.item():.6f}")

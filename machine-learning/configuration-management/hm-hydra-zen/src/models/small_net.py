from collections.abc import Callable

import torch
from torch import nn


class SmallNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: Callable[[], nn.Module],
    ) -> None:
        super().__init__()
        self.layer1 = nn.Linear(input_size, 32)
        self.activation = activation()
        self.layer2 = nn.Linear(32, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.activation(self.layer1(x)))

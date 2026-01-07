from collections.abc import Callable

import torch
from torch import nn


class LargeNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: Callable[[], nn.Module],
    ) -> None:
        super().__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.activation1 = activation()
        self.layer2 = nn.Linear(128, 64)
        self.activation2 = activation()
        self.layer3 = nn.Linear(64, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation1(self.layer1(x))
        x = self.activation2(self.layer2(x))
        return self.layer3(x)

import torch
from models.simple_mamba2 import SimpleMamba2
from torch import nn


class SimpleMamba2Block(nn.Module):
    """A complete Mamba 2 block with layer normalization and residual connection."""

    def __init__(
        self,
        dimension: int,
        state_dimension: int = 64,
        head_count: int = 8,
        expand_factor: int = 2,
    ) -> None:
        super().__init__()

        self.layer_norm = nn.LayerNorm(dimension)
        self.mamba = SimpleMamba2(
            dimension=dimension,
            state_dimension=state_dimension,
            head_count=head_count,
            expand_factor=expand_factor,
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        normalized = self.layer_norm(input_tensor)
        mamba_output = self.mamba(normalized)
        return input_tensor + mamba_output

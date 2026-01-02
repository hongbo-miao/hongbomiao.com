import torch
from models.simple_mamba2_block import SimpleMamba2Block
from torch import nn


class SelectiveCopyModel(nn.Module):
    """Small Mamba 2 model for selective copying task."""

    def __init__(
        self,
        vocabulary_size: int,
        embedding_dimension: int,
        state_dimension: int,
        head_count: int,
        layer_count: int,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension)
        self.mamba_layers = nn.ModuleList(
            [
                SimpleMamba2Block(
                    dimension=embedding_dimension,
                    state_dimension=state_dimension,
                    head_count=head_count,
                    expand_factor=2,
                )
                for _ in range(layer_count)
            ],
        )
        self.output_projection = nn.Linear(embedding_dimension, vocabulary_size)

    def forward(self, input_tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(input_tokens)

        for mamba_layer in self.mamba_layers:
            hidden = mamba_layer(hidden)

        return self.output_projection(hidden)

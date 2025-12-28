import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class Expert(nn.Module):
    """
    Expert Network: A simple feedforward network.

    Each expert is a 2-layer MLP that processes input tokens.
    In MoE, multiple experts exist, but only top-k are activated per token.

    Math:
        h = ReLU(W1 * x + b1)    # Hidden layer
        y = W2 * h + b2          # Output layer
    """

    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        output_dimension: int,
    ) -> None:
        super().__init__()
        # W1: [input_dimension, hidden_dimension]
        self.linear_1 = nn.Linear(input_dimension, hidden_dimension)
        # W2: [hidden_dimension, output_dimension]
        self.linear_2 = nn.Linear(hidden_dimension, output_dimension)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # h = ReLU(W1 * x + b1)
        hidden = F.relu(self.linear_1(input_tensor))
        # y = W2 * h + b2
        return self.linear_2(hidden)

import torch
from torch import nn


class FlowMatchingModel(nn.Module):
    """
    Neural network that predicts the velocity field v(x, t).

    The model takes a point x and time t, and outputs the predicted velocity
    at that point and time. This velocity tells us how to move x to transform
    the noise distribution into the data distribution.

    Dimensions:
        - position: (batch_size, input_dimension)
        - time: (batch_size, 1)
        - output velocity: (batch_size, input_dimension)

    Architecture: MLP with time embedding concatenated to input.
    """

    def __init__(self, input_dimension: int, hidden_dimension: int) -> None:
        super().__init__()
        self.input_dimension = input_dimension
        self.time_embedding = nn.Sequential(
            nn.Linear(1, hidden_dimension),
            nn.SiLU(),
            nn.Linear(hidden_dimension, hidden_dimension),
        )
        self.network = nn.Sequential(
            nn.Linear(input_dimension + hidden_dimension, hidden_dimension),
            nn.SiLU(),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.SiLU(),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.SiLU(),
            nn.Linear(hidden_dimension, input_dimension),
        )

    def forward(self, position: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        Predict velocity at given position and time.

        Args:
            position: Points in space, shape (batch_size, input_dimension)
            time: Time values in [0, 1], shape (batch_size, 1)

        Returns:
            Predicted velocity, shape (batch_size, input_dimension)

        """
        # (batch_size, hidden_dimension)
        time_features = self.time_embedding(time)
        # (batch_size, input_dimension + hidden_dimension)
        combined_input = torch.cat([position, time_features], dim=-1)
        # (batch_size, input_dimension)
        return self.network(combined_input)

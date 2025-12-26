import torch
from models.liquid_time_constant_cell import LiquidTimeConstantCell
from torch import nn


class LiquidNeuralNetwork(nn.Module):
    """
    Liquid Neural Network for time-series prediction.

    Processes sequences using LTC cells:
        h_0 = 0
        h_t = LTC(x_t, h_{t-1}, delta_t) for t = 1, ..., T
        y_t = W_o * h_t + b_o
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.liquid_cell = LiquidTimeConstantCell(input_size, hidden_size)

        # W_o: output projection
        # y_t = W_o * h_t + b_o
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        input_sequence: torch.Tensor,
        time_delta: float = 1.0,
    ) -> torch.Tensor:
        batch_size = input_sequence.size(0)
        sequence_length = input_sequence.size(1)

        # Initialize hidden state: h_0 = 0
        hidden_state = torch.zeros(
            batch_size,
            self.hidden_size,
            device=input_sequence.device,
        )

        output_list = []
        for time_step in range(sequence_length):
            # Get input at current time step: x_t
            input_at_time_step = input_sequence[:, time_step, :]

            # Apply LTC cell: h_t = LTC(x_t, h_{t-1}, delta_t)
            hidden_state = self.liquid_cell(
                input_at_time_step,
                hidden_state,
                time_delta,
            )

            # Compute output: y_t = W_o * h_t + b_o
            output_at_time_step = self.output_layer(hidden_state)
            output_list.append(output_at_time_step)

        return torch.stack(output_list, dim=1)

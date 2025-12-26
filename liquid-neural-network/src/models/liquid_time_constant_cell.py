import torch
from torch import nn


class LiquidTimeConstantCell(nn.Module):
    """
    Liquid Time-Constant (LTC) cell for continuous-time sequence modeling.

    The continuous ODE (cannot run directly on computers):
        dh/dt = -h/tau + f(x,h)/tau

    Computers are discrete machines with finite precision, so we discretize
    using the exponential Euler method:
        h_{t+1} = alpha * h_t + (1 - alpha) * h_tilde
        where alpha = exp(-delta_t / tau)
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # W_x: maps input x to hidden dimension
        # Used in: h_tilde = tanh(W_x * x + W_h * h)
        self.input_weight = nn.Linear(input_size, hidden_size)

        # W_h: maps hidden state h to hidden dimension
        # Used in: h_tilde = tanh(W_x * x + W_h * h)
        self.hidden_weight = nn.Linear(hidden_size, hidden_size)

        # W_tau: maps concatenated [x; h] to time constants
        # Used in: tau = sigmoid(W_tau * [x; h] + b_tau)
        self.time_constant_weight = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(
        self,
        input_tensor: torch.Tensor,
        hidden_state: torch.Tensor,
        time_delta: float = 1.0,
    ) -> torch.Tensor:
        # Step 1: Concatenate input and hidden state
        # [x; h]
        combined = torch.cat([input_tensor, hidden_state], dim=-1)

        # Step 2: Compute adaptive time constants
        # tau = sigma(W_tau * [x; h] + b_tau)
        time_constant = torch.sigmoid(self.time_constant_weight(combined))

        # Step 3: Compute candidate activation
        # h_tilde = tanh(W_x * x + W_h * h + b)
        input_activation = torch.tanh(
            self.input_weight(input_tensor) + self.hidden_weight(hidden_state),
        )

        # Step 4: Compute decay factor
        # alpha = exp(-delta_t / tau)
        decay_factor = torch.exp(-time_delta / (time_constant + 1e-8))

        # Step 5: Blend old state and candidate activation
        # h_{t+1} = alpha * h_t + (1 - alpha) * h_tilde
        return decay_factor * hidden_state + (1 - decay_factor) * input_activation

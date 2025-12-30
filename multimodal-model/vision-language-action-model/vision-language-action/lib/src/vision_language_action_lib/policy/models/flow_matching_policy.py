import logging

import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)


class FlowMatchingPolicy(nn.Module):
    """Flow Matching policy for action generation."""

    def __init__(
        self,
        context_dimension: int = 3584,
        action_dimension: int = 6,
        hidden_dimension: int = 1024,
        layer_count: int = 6,
    ) -> None:
        super().__init__()

        self.context_dimension = context_dimension
        self.action_dimension = action_dimension

        input_dimension = context_dimension + action_dimension + 1

        layers = []
        layers.append(nn.Linear(input_dimension, hidden_dimension))
        layers.append(nn.SiLU())

        for _ in range(layer_count - 1):
            layers.append(nn.Linear(hidden_dimension, hidden_dimension))
            layers.append(nn.SiLU())

        layers.append(nn.Linear(hidden_dimension, action_dimension))

        self.vector_field_network = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        context: Tensor,
        noisy_action: Tensor,
        time: Tensor,
    ) -> Tensor:
        input_tensor = torch.cat([context, noisy_action, time], dim=-1)
        return self.vector_field_network(input_tensor)

    def compute_loss(
        self,
        context: Tensor,
        target_action: Tensor,
    ) -> Tensor:
        batch_size = context.shape[0]
        device = context.device

        time = torch.rand(batch_size, 1, device=device)

        noise = torch.randn_like(target_action)

        noisy_action = (1 - time) * noise + time * target_action

        target_velocity = target_action - noise

        predicted_velocity = self.forward(context, noisy_action, time)

        return nn.functional.mse_loss(predicted_velocity, target_velocity)

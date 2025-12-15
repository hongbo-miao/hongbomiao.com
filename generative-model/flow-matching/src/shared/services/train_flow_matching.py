import logging

import torch
from shared.services.flow_matching_model import FlowMatchingModel
from torch import nn

logger = logging.getLogger(__name__)


def train_flow_matching(
    model: FlowMatchingModel,
    target_data: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    epoch_count: int,
    batch_size: int,
) -> None:
    """
    Train the flow matching model.

    Training objective:
    - Sample random time t ~ Uniform(0, 1)
    - Sample noise x_0 ~ N(0, I)
    - Sample data x_1 from target distribution
    - Compute interpolated point: x_t = (1 - t)*x_0 + t*x_1
    - True velocity is: v_target = x_1 - x_0 (derivative of x_t w.r.t. t)
    - Train network to predict v_target given (x_t, t)

    This is called "Conditional Flow Matching" because we condition on
    specific pairs (x_0, x_1) rather than marginal distributions.
    """
    model.train()

    # target_data: (sample_count, input_dimension)
    sample_count = target_data.shape[0]
    device = target_data.device

    for epoch_index in range(epoch_count):
        # Sample a minibatch of target data points x_1.
        batch_indices = torch.randint(0, sample_count, (batch_size,), device=device)
        # x_1, (batch_size, input_dimension)
        data_samples = target_data[batch_indices]
        # Sample matching noise points x_0 with the same shape as x_1.
        # x_0, (batch_size, input_dimension)
        noise_samples = torch.randn_like(data_samples)

        # Sample time t for each point.
        time_values = torch.rand(batch_size, 1, device=device)  # (batch_size, 1)

        # Compute the interpolated points x_t along the straight path from x_0 to x_1.
        # x_t = (1 - t) * x_0 + t * x_1
        # x_t, interpolated_points: (batch_size, input_dimension)
        interpolated_points = (
            1 - time_values
        ) * noise_samples + time_values * data_samples

        # The true velocity along this path is dx_t/dt = x_1 - x_0.
        # v, (batch_size, input_dimension)
        true_velocity = data_samples - noise_samples
        # v_hat, (batch_size, input_dimension)
        predicted_velocity = model(
            interpolated_points,
            time_values,
        )

        # predicted_velocity: (batch_size, input_dimension)
        # true_velocity: (batch_size, input_dimension)
        loss = loss_function(predicted_velocity, true_velocity)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch_index + 1) % 200 == 0:
            logger.info(
                f"Epoch {epoch_index + 1}/{epoch_count}, Loss: {loss.item():.6f}",
            )

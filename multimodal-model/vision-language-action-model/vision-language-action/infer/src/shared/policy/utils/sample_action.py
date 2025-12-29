import logging

import torch
from torch import Tensor
from vision_language_action_lib.policy.models.flow_matching_policy import (
    FlowMatchingPolicy,
)
from vision_language_action_lib.types.action_output import ActionOutput

logger = logging.getLogger(__name__)


def sample_action(
    policy: FlowMatchingPolicy,
    context: Tensor,
    num_integration_steps: int = 20,
    integration_method: str = "euler",
    seed: int | None = None,
) -> ActionOutput:
    """
    Sample action from Flow Matching policy via ODE integration.

    Args:
        policy: Trained FlowMatchingPolicy model
        context: Context embedding from vision-language fusion [1, context_dim]
        num_integration_steps: Number of ODE integration steps
        integration_method: Integration method ("euler" or "midpoint")
        seed: Random seed for reproducibility

    Returns:
        ActionOutput with 6-DoF action values

    """
    device = context.device
    dtype = context.dtype
    batch_size = context.shape[0]

    if seed is not None:
        torch.manual_seed(seed)

    action = torch.randn(
        batch_size,
        policy.action_dimension,
        device=device,
        dtype=dtype,
    )

    delta_time = 1.0 / num_integration_steps

    policy.eval()
    with torch.no_grad():
        for step in range(num_integration_steps):
            current_time = torch.full(
                (batch_size, 1),
                step / num_integration_steps,
                device=device,
                dtype=dtype,
            )

            if integration_method == "euler":
                velocity = policy(context, action, current_time)
                action = action + velocity * delta_time

            elif integration_method == "midpoint":
                velocity_start = policy(context, action, current_time)
                action_midpoint = action + velocity_start * (delta_time / 2)
                time_midpoint = current_time + delta_time / 2
                velocity_midpoint = policy(context, action_midpoint, time_midpoint)
                action = action + velocity_midpoint * delta_time

            else:
                msg = f"Unknown integration method: {integration_method}"
                raise ValueError(msg)

    action = torch.clamp(action, min=-2.0, max=2.0)

    action_values = action.squeeze(0).cpu().tolist()

    logger.debug(f"Sampled action: {action_values}")

    return ActionOutput(
        delta_x=action_values[0],
        delta_y=action_values[1],
        delta_z=action_values[2],
        delta_roll=action_values[3],
        delta_pitch=action_values[4],
        delta_yaw=action_values[5],
    )

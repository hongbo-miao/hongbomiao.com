import torch
from shared.services.flow_matching_model import FlowMatchingModel


@torch.no_grad()
def generate_samples(
    model: FlowMatchingModel,
    sample_count: int,
    time_step_count: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate samples by integrating the learned velocity field.

    We use Euler integration: x_{t+dt} = x_t + v(x_t, t) * dt
    Starting from Gaussian noise at t=0, we integrate to t=1 to get samples.
    """
    model.eval()

    samples = torch.randn(sample_count, model.input_dimension, device=device)
    time_step_size = 1.0 / time_step_count

    for step_index in range(time_step_count):
        current_time = step_index / time_step_count
        time_tensor = torch.full((sample_count, 1), current_time, device=device)

        velocity = model(samples, time_tensor)
        samples = samples + velocity * time_step_size

    return samples

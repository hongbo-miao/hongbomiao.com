import math

import torch


def generate_sine_wave_data(
    sample_count: int,
    sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    time_values = torch.linspace(0, 4 * math.pi, sample_count + sequence_length)
    sine_values = torch.sin(time_values)
    sequences = sine_values.unfold(0, sequence_length + 1, 1)
    input_tensor = sequences[:, :-1].unsqueeze(-1)
    target_tensor = sequences[:, 1:].unsqueeze(-1)
    return input_tensor, target_tensor

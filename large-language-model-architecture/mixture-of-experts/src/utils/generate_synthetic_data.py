import torch


def generate_synthetic_data(
    sample_count: int,
    sequence_length: int,
    input_dimension: int,
    target_transform: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_data = torch.randn(sample_count, sequence_length, input_dimension)
    target_data = torch.tanh(input_data @ target_transform)
    return input_data, target_data

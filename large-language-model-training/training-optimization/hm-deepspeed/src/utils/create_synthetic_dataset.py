import torch
from torch.utils.data import TensorDataset


def create_synthetic_dataset(
    sample_count: int,
    input_size: int,
    output_size: int,
) -> TensorDataset:
    input_data = torch.randn(sample_count, input_size)
    target_data = torch.randn(sample_count, output_size)
    return TensorDataset(input_data, target_data)

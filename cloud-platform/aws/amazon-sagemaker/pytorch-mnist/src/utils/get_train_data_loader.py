import logging
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)


def get_train_data_loader(
    batch_size: int,
    training_dir: str,
    is_distributed: bool,
    **kwargs: Any,  # noqa: ANN401
) -> DataLoader:
    logger.info("Get train data loader")
    dataset: Dataset = datasets.MNIST(
        training_dir,
        train=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))],
        ),
    )

    train_sampler: DistributedSampler | None = (
        DistributedSampler(dataset) if is_distributed else None
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        **kwargs,
    )

import logging
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)


def get_test_data_loader(
    test_batch_size: int,
    training_dir: str,
    **kwargs: Any,  # noqa: ANN401
) -> DataLoader:
    logger.info("Get test data loader")
    return torch.utils.data.DataLoader(
        datasets.MNIST(
            training_dir,
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))],
            ),
        ),
        batch_size=test_batch_size,
        shuffle=True,
        **kwargs,
    )

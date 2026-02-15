import logging

import torch
import torch.nn.functional as F  # noqa: N812
from torch.nn import Module
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def test(model: Module, test_loader: DataLoader, device: str | torch.device) -> None:
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            device_data, device_target = data.to(device), target.to(device)
            output = model(device_data)
            # sum up batch loss
            test_loss += F.nll_loss(output, device_target, size_average=False).item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(device_target.view_as(pred)).sum().item()

    dataset_len: int = len(test_loader.dataset)  # type: ignore[arg-type]
    test_loss /= dataset_len
    logger.info(
        f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{dataset_len} ({100.0 * correct / dataset_len:.0f}%)\n",
    )

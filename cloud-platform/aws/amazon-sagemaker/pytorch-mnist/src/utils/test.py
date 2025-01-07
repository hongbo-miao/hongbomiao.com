import logging

import torch
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data.distributed

logger = logging.getLogger(__name__)


def test(model, test_loader, device):
    model.eval()
    test_loss = 0
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

    test_loss /= len(test_loader.dataset)
    logger.info(
        f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100.0 * correct / len(test_loader.dataset):.0f}%)\n",
    )

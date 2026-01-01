import torch
from torch import nn
from torch.utils.data import DataLoader


def evaluate(net: nn.Module, data_loader: DataLoader, device: str) -> float:
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net(inputs)
            # The class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def train(
    net: nn.Module,
    data_loader: DataLoader,
    device: str,
    optimizer: Optimizer,
    criterion: nn.Module,
) -> float:
    net.train()
    running_loss = 0.0

    for i, data in enumerate(data_loader, 0):
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss

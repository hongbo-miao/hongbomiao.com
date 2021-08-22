import torch

from model.data_loader import val_data_loader
from model.net import net
from utils.device import device


def evaluate(
    net,
    val_data_loader,
):
    net.eval()
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in val_data_loader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


if __name__ == "__main__":
    net.to(device)
    val_acc = evaluate(net, val_data_loader)
    print(
        "Accuracy of the network on the 10000 validation images: %d %%"
        % (100 * val_acc)
    )

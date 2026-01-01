import torch
import torchvision
from args import get_args
from torchvision import transforms

args = get_args()
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))],
)
batch_size = 4

train_set = torchvision.datasets.CIFAR10(
    root="./data/processed",
    train=True,
    download=args.should_download_original_data,
    transform=transform,
)
train_data_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
)
val_set = torchvision.datasets.CIFAR10(
    root="./data/processed",
    train=False,
    download=args.should_download_original_data,
    transform=transform,
)
val_data_loader = torch.utils.data.DataLoader(
    val_set,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
)

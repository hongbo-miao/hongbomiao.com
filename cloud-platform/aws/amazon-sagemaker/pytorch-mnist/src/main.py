import os

import torch
import torch.utils.data
import torch.utils.data.distributed
from models.net import Net
from utils.get_args import get_args
from utils.train import train


def model_fn(model_dir: str) -> Net:
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model = torch.nn.DataParallel(Net())
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)


if __name__ == "__main__":
    args = get_args()
    train(args)

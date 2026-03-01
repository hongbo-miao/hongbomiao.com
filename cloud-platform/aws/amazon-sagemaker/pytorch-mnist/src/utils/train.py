import argparse
import logging
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F  # noqa: N812
import torch.utils.data
import torch.utils.data.distributed
from models.net import Net
from torch import nn, optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from utils.average_gradients import average_gradients
from utils.get_test_data_loader import get_test_data_loader
from utils.get_train_data_loader import get_train_data_loader
from utils.save_model import save_model
from utils.test import test

logger = logging.getLogger(__name__)


def _setup_distributed(args: argparse.Namespace) -> None:
    world_size = len(args.hosts)
    os.environ["WORLD_SIZE"] = str(world_size)
    host_rank = args.hosts.index(args.current_host)
    os.environ["RANK"] = str(host_rank)
    dist.init_process_group(
        backend=args.backend,
        rank=host_rank,
        world_size=world_size,
    )
    logger.info(
        f"Initialized the distributed environment: '{args.backend}' backend on {dist.get_world_size()} nodes. "
        f"Current host rank is {dist.get_rank()}. Number of gpus: {args.num_gpus}",
    )


def _setup_model(
    args: argparse.Namespace,
    device: torch.device,
    is_distributed: bool,
    use_cuda: bool,
) -> tuple[nn.Module, Optimizer]:
    model: nn.Module = Net().to(device)
    if is_distributed and use_cuda:
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        model = torch.nn.DataParallel(model)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    return model, optimizer


def _train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    is_distributed: bool,
    use_cuda: bool,
    log_interval: int,
    epoch: int,
) -> None:
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader, 1):
        device_data, device_target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(device_data)
        loss = F.nll_loss(output, device_target)
        loss.backward()
        if is_distributed and not use_cuda:
            average_gradients(model)
        optimizer.step()
        if batch_idx % log_interval == 0:
            logger.info(
                f"Train Epoch: {epoch} [{batch_idx * len(device_data)}/{len(train_loader.sampler)} "
                f"({100.0 * batch_idx / len(train_loader):.0f}%)] Loss: {loss.item():.6f}",
            )


def train(args: argparse.Namespace) -> None:
    is_distributed = len(args.hosts) > 1 and args.backend is not None
    logger.info(f"Distributed training: {is_distributed}")

    use_cuda = args.num_gpus > 0
    logger.info(f"Number of gpus available: {args.num_gpus}")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu",
    )

    if is_distributed:
        _setup_distributed(args)

    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    train_loader = get_train_data_loader(
        args.batch_size,
        args.data_dir,
        is_distributed,
        **kwargs,
    )
    test_loader = get_test_data_loader(args.test_batch_size, args.data_dir, **kwargs)

    logger.info(
        f"Processes {len(train_loader.sampler)}/{len(train_loader.dataset)} ({100.0 * len(train_loader.sampler) / len(train_loader.dataset):.0f}%) of train data",
    )
    logger.info(
        f"Processes {len(test_loader.sampler)}/{len(test_loader.dataset)} ({100.0 * len(test_loader.sampler) / len(test_loader.dataset):.0f}%) of test data",
    )

    model, optimizer = _setup_model(args, device, is_distributed, use_cuda)

    for epoch in range(1, args.epochs + 1):
        _train_one_epoch(model, train_loader, optimizer, device, is_distributed, use_cuda, args.log_interval, epoch)
        test(model, test_loader, device)

    save_model(model, Path(args.model_dir))

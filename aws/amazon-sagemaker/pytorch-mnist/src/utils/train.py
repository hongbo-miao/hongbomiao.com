import logging
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from models.net import Net
from utils.average_gradients import average_gradients
from utils.get_test_data_loader import get_test_data_loader
from utils.get_train_data_loader import get_train_data_loader
from utils.save_model import save_model
from utils.test import test


def train(args):
    is_distributed = len(args.hosts) > 1 and args.backend is not None
    logging.info("Distributed training:", is_distributed)

    use_cuda = args.num_gpus > 0
    logging.info("Number of gpus available:", args.num_gpus)
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")

    if is_distributed:
        # Initialize the distributed environment.
        world_size = len(args.hosts)
        os.environ["WORLD_SIZE"] = str(world_size)
        host_rank = args.hosts.index(args.current_host)
        os.environ["RANK"] = str(host_rank)
        dist.init_process_group(
            backend=args.backend, rank=host_rank, world_size=world_size
        )
        logging.info(
            "Initialized the distributed environment: '{}' backend on {} nodes. ".format(
                args.backend, dist.get_world_size()
            )
            + "Current host rank is {}. Number of gpus: {}".format(
                dist.get_rank(), args.num_gpus
            )
        )

    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    train_loader = get_train_data_loader(
        args.batch_size, args.data_dir, is_distributed, **kwargs
    )
    test_loader = get_test_data_loader(args.test_batch_size, args.data_dir, **kwargs)

    logging.info(
        "Processes {}/{} ({:.0f}%) of train data".format(
            len(train_loader.sampler),
            len(train_loader.dataset),
            100.0 * len(train_loader.sampler) / len(train_loader.dataset),
        )
    )

    logging.info(
        "Processes {}/{} ({:.0f}%) of test data".format(
            len(test_loader.sampler),
            len(test_loader.dataset),
            100.0 * len(test_loader.sampler) / len(test_loader.dataset),
        )
    )

    model = Net().to(device)
    if is_distributed and use_cuda:
        # multi-machine multi-gpu case
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        # single-machine multi-gpu case or single-machine or multi-machine cpu case
        model = torch.nn.DataParallel(model)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader, 1):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            if is_distributed and not use_cuda:
                # average gradients manually for a multi-machine cpu case only
                average_gradients(model)
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                logging.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.sampler),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
        test(model, test_loader, device)
    save_model(model, args.model_dir)

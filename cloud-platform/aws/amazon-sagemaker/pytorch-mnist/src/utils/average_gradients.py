import torch
import torch.distributed as dist


def average_gradients(model: torch.nn.Module) -> None:
    # Gradient averaging.
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size

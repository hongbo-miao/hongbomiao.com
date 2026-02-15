import torch
import torch.distributed as dist


def average_gradients(model: torch.nn.Module) -> None:
    # Gradient averaging.
    size = float(dist.get_world_size())  # type: ignore[possibly-missing-attribute]
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)  # type: ignore[possibly-missing-attribute]
            param.grad.data /= size

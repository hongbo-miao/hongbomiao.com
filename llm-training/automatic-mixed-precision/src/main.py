# https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html

import gc
import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


@contextmanager
def timer_context(message: str) -> Generator[None]:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_time: float = time.time()

    try:
        yield
    finally:
        torch.cuda.synchronize()
        end_time: float = time.time()
        logger.info(message)
        logger.info(f"Total execution time = {end_time - start_time:.3f} sec")
        logger.info(
            f"Max memory used by tensors = {torch.cuda.max_memory_allocated()} bytes",
        )


def create_model(
    input_size: int,
    output_size: int,
    num_layers: int,
) -> torch.nn.Sequential:
    layers: list[torch.nn.Module] = []
    for _ in range(num_layers - 1):
        layers.append(torch.nn.Linear(input_size, input_size))
        layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Linear(input_size, output_size))
    return torch.nn.Sequential(*tuple(layers)).cuda()


def train_default_precision(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    data: list[torch.Tensor],
    targets: list[torch.Tensor],
    epochs: int,
) -> None:
    logger.info("Starting default precision (float32) training...")
    with timer_context("Default precision:"):
        for _epoch in range(epochs):
            for input_tensor, target in zip(data, targets, strict=True):
                output: torch.Tensor = model(input_tensor)
                loss: torch.Tensor = loss_fn(output, target)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()


def train_mixed_precision(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    data: list[torch.Tensor],
    targets: list[torch.Tensor],
    epochs: int,
    device: str,
    use_amp: bool = True,
) -> torch.amp.GradScaler:
    logger.info("Starting mixed precision training...")
    scaler: torch.amp.GradScaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    with timer_context("Mixed precision:"):
        for _epoch in range(epochs):
            for input_tensor, target in zip(data, targets, strict=True):
                with torch.autocast(
                    device_type=device,
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    output: torch.Tensor = model(input_tensor)
                    loss: torch.Tensor = loss_fn(output, target)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

    return scaler


def demonstrate_autocast_behavior(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    data: list[torch.Tensor],
    targets: list[torch.Tensor],
    device: str,
) -> None:
    logger.info("Demonstrating autocast behavior...")

    input_tensor: torch.Tensor = data[0]
    target: torch.Tensor = targets[0]

    with torch.autocast(device_type=device, dtype=torch.float16):
        output: torch.Tensor = model(input_tensor)
        # Output is float16 because linear layers autocast to float16
        logger.info(f"Output dtype with autocast: {output.dtype}")

        loss: torch.Tensor = loss_fn(output, target)
        # Loss is float32 because mse_loss autocasts to float32
        logger.info(f"Loss dtype with autocast: {loss.dtype}")


def demonstrate_gradient_clipping(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    data: list[torch.Tensor],
    targets: list[torch.Tensor],
    device: str,
) -> None:
    logger.info("Demonstrating gradient clipping with mixed precision...")
    scaler: torch.amp.GradScaler = torch.amp.GradScaler("cuda")

    # Run for just one batch as demonstration
    input_tensor: torch.Tensor = data[0]
    target: torch.Tensor = targets[0]

    with torch.autocast(device_type=device, dtype=torch.float16):
        output: torch.Tensor = model(input_tensor)
        loss: torch.Tensor = loss_fn(output, target)

    scaler.scale(loss).backward()

    # Unscale gradients before clipping
    scaler.unscale_(optimizer)

    # Clip gradients as usual
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

    logger.info("Gradient clipping demonstration completed")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    filename: Path,
) -> dict[str, Any]:
    checkpoint: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
    }
    torch.save(checkpoint, filename)
    logger.info(f"Checkpoint saved to {filename}")
    return checkpoint


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    checkpoint: dict[str, Any],
) -> None:
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])
    logger.info("Checkpoint loaded successfully")


def demonstrate_inference_with_autocast(
    model: torch.nn.Module,
    data: list[torch.Tensor],
    device: str,
) -> None:
    logger.info("Demonstrating inference with autocast...")

    model.eval()
    with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.float16):
        # Run inference on first batch
        output: torch.Tensor = model(data[0])
        logger.info(f"Inference output shape: {output.shape}")
        logger.info(f"Inference output dtype: {output.dtype}")

    model.train()


def run_performance_comparison(
    batch_size: int,
    input_size: int,
    output_size: int,
    num_layers: int,
    num_batches: int,
    epochs: int,
    device: str,
) -> tuple[
    torch.nn.Module,
    torch.optim.Optimizer,
    torch.amp.GradScaler,
    torch.nn.Module,
    list[torch.Tensor],
    list[torch.Tensor],
]:
    logger.info("=" * 50)
    logger.info("PERFORMANCE COMPARISON")

    # Create synthetic data
    logger.info("Creating synthetic data...")
    data: list[torch.Tensor] = [
        torch.randn(batch_size, input_size) for _ in range(num_batches)
    ]
    targets: list[torch.Tensor] = [
        torch.randn(batch_size, output_size) for _ in range(num_batches)
    ]
    loss_fn: torch.nn.MSELoss = torch.nn.MSELoss().cuda()

    # Train with default precision
    logger.info("=" * 50)
    logger.info("TRAINING WITH DEFAULT PRECISION")

    model: torch.nn.Module = create_model(input_size, output_size, num_layers)
    optimizer: torch.optim.SGD = torch.optim.SGD(model.parameters(), lr=0.001)
    train_default_precision(model, optimizer, loss_fn, data, targets, epochs)

    # Train with mixed precision
    logger.info("=" * 50)
    logger.info("TRAINING WITH MIXED PRECISION")

    model = create_model(input_size, output_size, num_layers)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    scaler: torch.amp.GradScaler = train_mixed_precision(
        model,
        optimizer,
        loss_fn,
        data,
        targets,
        epochs,
        device,
        use_amp=True,
    )

    return model, optimizer, scaler, loss_fn, data, targets


def demonstrate_advanced_features(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    loss_fn: torch.nn.Module,
    data: list[torch.Tensor],
    targets: list[torch.Tensor],
    device: str,
) -> None:
    logger.info("=" * 50)
    logger.info("ADVANCED FEATURES DEMONSTRATION")

    # Autocast behavior demonstration
    demonstrate_autocast_behavior(model, loss_fn, data, targets, device)

    # Gradient clipping demonstration
    demonstrate_gradient_clipping(model, optimizer, loss_fn, data, targets, device)

    # Inference demonstration
    demonstrate_inference_with_autocast(model, data, device)

    # Checkpoint saving/loading demonstration
    checkpoint: dict[str, Any] = save_checkpoint(
        model,
        optimizer,
        scaler,
        Path("checkpoint.pt"),
    )

    # Create new instances and load checkpoint
    new_net: torch.nn.Module = create_model(data[0].shape[1], targets[0].shape[1], 3)
    new_opt: torch.optim.SGD = torch.optim.SGD(new_net.parameters(), lr=0.001)
    new_scaler: torch.amp.GradScaler = torch.amp.GradScaler("cuda")

    load_checkpoint(new_net, new_opt, new_scaler, checkpoint)


def main() -> None:
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This example requires a CUDA-capable GPU.")
        return

    batch_size: int = 512
    input_size: int = 4096
    output_size: int = 4096
    num_layers: int = 3
    num_batches: int = 50
    epochs: int = 10
    device: str = "cuda"

    logger.info("Configuration:")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Input size: {input_size}")
    logger.info(f"  Output size: {output_size}")
    logger.info(f"  Number of layers: {num_layers}")
    logger.info(f"  Number of batches: {num_batches}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Device: {device}")

    torch.set_default_device(device)

    model, optimizer, scaler, loss_fn, data, targets = run_performance_comparison(
        batch_size,
        input_size,
        output_size,
        num_layers,
        num_batches,
        epochs,
        device,
    )

    demonstrate_advanced_features(
        model,
        optimizer,
        scaler,
        loss_fn,
        data,
        targets,
        device,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()

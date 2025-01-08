import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def vector_add_kernel(
    x_ptr,  # Pointer to first vector  # noqa: ANN001
    y_ptr,  # Pointer to second vector  # noqa: ANN001
    output_ptr,  # Pointer to output vector  # noqa: ANN001
    n_elements: int,  # Number of elements in the vectors
    block_size: tl.constexpr,  # Number of elements each program should process
) -> None:
    # Program ID
    pid = tl.program_id(axis=0)

    # Calculate the start index for this program instance
    block_start = pid * block_size

    # Create an offset array for this block
    offsets = block_start + tl.arange(0, block_size)

    # Create a mask to handle the case where array size isn't multiple of block_size
    mask = offsets < n_elements

    # Load data using the mask
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Perform the addition
    output = x + y

    # Store the result using the same mask
    tl.store(output_ptr + offsets, output, mask=mask)


def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    n_elements: int = x.shape[0]

    # Allocate output
    output: torch.Tensor = torch.empty_like(x)

    # Define block size (can be tuned for performance)
    block_size: int = 128

    # Calculate grid size
    grid: tuple[int, ...] = (triton.cdiv(n_elements, block_size),)

    # Launch kernel
    vector_add_kernel[grid](
        x,  # Triton automatically converts tensor to pointer
        y,  # Triton automatically converts tensor to pointer
        output,  # Triton automatically converts tensor to pointer
        n_elements,
        block_size,
    )

    return output


def main() -> None:
    # Create input vectors
    size: int = 1024
    x: torch.Tensor = torch.randn(size, device="cuda")
    y: torch.Tensor = torch.randn(size, device="cuda")

    # Run Triton kernel
    output_triton: torch.Tensor = vector_add(x, y)

    # Verify result against PyTorch
    output_torch: torch.Tensor = x + y
    logger.info(output_torch)
    logger.info(output_triton)
    logger.info(f"Max difference: {torch.max(torch.abs(output_torch - output_triton))}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()

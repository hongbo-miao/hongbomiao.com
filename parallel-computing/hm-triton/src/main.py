import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def vector_add_kernel(
    x_ptr,  # Pointer to first vector
    y_ptr,  # Pointer to second vector
    output_ptr,  # Pointer to output vector
    n_elements,  # Number of elements in the vectors
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process
) -> None:
    # Program ID
    pid = tl.program_id(axis=0)

    # Calculate the start index for this program instance
    block_start = pid * BLOCK_SIZE

    # Create an offset array for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Create a mask to handle the case where array size isn't multiple of BLOCK_SIZE
    mask = offsets < n_elements

    # Load data using the mask
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Perform the addition
    output = x + y

    # Store the result using the same mask
    tl.store(output_ptr + offsets, output, mask=mask)


def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Assert vectors are same size and on correct device
    assert x.shape == y.shape
    assert x.is_cuda and y.is_cuda
    n_elements: int = x.shape[0]

    # Allocate output
    output: torch.Tensor = torch.empty_like(x)

    # Define block size (can be tuned for performance)
    BLOCK_SIZE: int = 128

    # Calculate grid size
    grid: tuple[int, ...] = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Launch kernel
    vector_add_kernel[grid](
        x,  # Triton automatically converts tensor to pointer
        y,  # Triton automatically converts tensor to pointer
        output,  # Triton automatically converts tensor to pointer
        n_elements,
        BLOCK_SIZE,
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

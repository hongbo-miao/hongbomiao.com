import logging
import time

import numba as nb
import numpy as np
from numba import cuda

logger = logging.getLogger(__name__)


def sum_of_squares_python(n: int) -> float:
    total = 0.0
    for i in range(n):
        total += i * i
    return total


@nb.jit(nopython=True)
def sum_of_squares_cpu_numba(n: int) -> float:
    total: float = 0.0
    for i in range(n):
        total += i * i
    return total


@cuda.jit
def sum_of_squares_cuda_kernel(n: int, result: cuda.devicearray.DeviceNDArray) -> None:
    idx = cuda.grid(1)
    if idx < n:
        cuda.atomic.add(result, 0, idx * idx)


def sum_of_squares_gpu_numba(n: int) -> float:
    # Allocate memory on GPU
    result = cuda.device_array(1, dtype=np.float64)
    cuda.to_device(np.array([0.0], dtype=np.float64), to=result)

    # Configure the blocks
    threads_per_block = 256
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

    # Run the kernel
    sum_of_squares_cuda_kernel[blocks_per_grid, threads_per_block](n, result)

    # Copy result back to CPU and return
    return result.copy_to_host()[0]


def main() -> None:
    n = 10_000_000

    # Python version
    start_time = time.time()
    python_result = sum_of_squares_python(n)
    python_time = time.time() - start_time
    logger.info(f"Python result: {python_result}")
    logger.info(f"Python time: {python_time} seconds")

    # CPU Numba version
    start_time = time.time()
    cpu_numba_result = sum_of_squares_cpu_numba(n)
    cpu_numba_time = time.time() - start_time
    logger.info(f"CPU Numba result: {cpu_numba_result}")
    logger.info(f"CPU Numba time: {cpu_numba_time} seconds")
    logger.info(f"CPU Numba speedup vs Python: {python_time / cpu_numba_time}x")

    # GPU Numba version
    try:
        start_time = time.time()
        gpu_numba_result = sum_of_squares_gpu_numba(n)
        gpu_numba_time = time.time() - start_time
        logger.info(f"GPU Numba result: {gpu_numba_result}")
        logger.info(f"GPU Numba time: {gpu_numba_time} seconds")
        logger.info(f"GPU Numba speedup vs Python: {python_time / gpu_numba_time}x")
    except cuda.CudaSupportError:
        logger.warning("CUDA GPU is not available")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()

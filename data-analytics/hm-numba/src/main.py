import logging
import time

import numba as nb
import numpy as np
from numba import cuda


@nb.jit(nopython=True)
def sum_of_squares_numba(n: int) -> float:
    total: float = 0.0
    for i in range(n):
        total += i * i
    return total


@cuda.jit
def sum_of_squares_cuda_kernel(n: int, result: cuda.devicearray.DeviceNDArray) -> None:
    idx = cuda.grid(1)
    if idx < n:
        cuda.atomic.add(result, 0, idx * idx)


def sum_of_squares_gpu(n: int) -> float:
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


def sum_of_squares_python(n: int) -> float:
    total = 0.0
    for i in range(n):
        total += i * i
    return total


def main() -> None:
    n = 10_000_000

    # Python version
    start_time = time.time()
    result_python = sum_of_squares_python(n)
    python_time = time.time() - start_time
    logging.info(f"Python result: {result_python}")
    logging.info(f"Python time: {python_time} seconds")

    # CPU Numba version
    start_time = time.time()
    result_numba = sum_of_squares_numba(n)
    numba_time = time.time() - start_time
    logging.info(f"CPU Numba result: {result_numba}")
    logging.info(f"CPU Numba time: {numba_time} seconds")
    logging.info(f"CPU Numba speedup vs Python: {python_time / numba_time}x")

    # GPU version
    try:
        start_time = time.time()
        result_gpu = sum_of_squares_gpu(n)
        gpu_time = time.time() - start_time
        logging.info(f"GPU result: {result_gpu}")
        logging.info(f"GPU time: {gpu_time} seconds")
        logging.info(f"GPU speedup vs Python: {python_time / gpu_time}x")
    except cuda.CudaSupportError:
        logging.warning("CUDA GPU is not available")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

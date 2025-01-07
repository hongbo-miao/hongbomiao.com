import logging
import time

import cupy as cp
import numpy as np

logger = logging.getLogger(__name__)


def sum_of_squares_numpy(n: int) -> float:
    array = np.arange(n)
    return np.sum(array * array)


def sum_of_squares_cupy(n: int) -> float:
    array = cp.arange(n)
    return float(cp.sum(array * array).get())


def main() -> None:
    n = 10_000_000

    # NumPy version
    start_time = time.time()
    result_numpy = sum_of_squares_numpy(n)
    numpy_time = time.time() - start_time
    logger.info(f"NumPy result: {result_numpy}")
    logger.info(f"NumPy time: {numpy_time} seconds")

    try:
        # CuPy version
        start_time = time.time()
        result_cupy = sum_of_squares_cupy(n)
        cupy_time = time.time() - start_time
        logger.info(f"CuPy result: {result_cupy}")
        logger.info(f"CuPy time: {cupy_time} seconds")
        logger.info(f"CuPy speedup vs NumPy: {numpy_time / cupy_time}x")

    except cp.cuda.runtime.CUDARuntimeError:
        logger.warning("CUDA GPU is not available")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()

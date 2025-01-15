import logging
import time

import taichi as ti

logger = logging.getLogger(__name__)

ti.init(arch=ti.gpu)


# Taichi implementation
@ti.func
def is_prime_taichi(n: int) -> int:
    result = 1

    if n <= 1:
        result = 0
    if n == 2:
        result = 1
    if n % 2 == 0 and n > 2:
        result = 0

    i = 3
    while i * i <= n:
        if n % i == 0:
            result = 0
        i += 2
    return result


@ti.kernel
def count_primes_taichi(n: int) -> int:
    count = 0
    for i in range(n + 1):
        count += is_prime_taichi(i)
    return count


# Python implementation
def is_prime_python(n: int) -> bool:
    if n <= 1:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def count_primes_python(n: int) -> int:
    count = 0
    for i in range(n + 1):
        if is_prime_python(i):
            count += 1
    return count


def main() -> None:
    numbers = [10, 100, 1000, 10000, 100000, 1000000]

    # Test Python implementation
    logger.info("Python:")
    logger.info("Number\tCount\tTime (ms)")

    for n in numbers:
        start_time = time.time()
        result = count_primes_python(n)
        end_time = time.time()

        execution_time = (end_time - start_time) * 1000
        logger.info(f"{n}\t{result}\t{execution_time:.2f}")

    # Test Taichi implementation
    logger.info("-" * 40)
    logger.info("Taichi:")
    logger.info("Number\tCount\tTime (ms)")

    for n in numbers:
        start_time = time.time()
        result = count_primes_taichi(n)
        end_time = time.time()

        execution_time = (end_time - start_time) * 1000
        logger.info(f"{n}\t{result}\t{execution_time:.2f}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()

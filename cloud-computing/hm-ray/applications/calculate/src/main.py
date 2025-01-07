import logging
import time

import ray

logger = logging.getLogger(__name__)


@ray.remote
def square(n: int) -> int:
    time.sleep(5)
    return n * n


@ray.remote
def sum_list(numbers: list[int]) -> int:
    return sum(numbers)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    ray.init()
    logger.info(ray.cluster_resources())

    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    squared_tasks = [square.remote(n) for n in numbers]
    squared_results: list[int] = ray.get(squared_tasks)
    logger.info(f"{squared_results = }")

    sum_task = sum_list.remote(squared_results)
    total_sum = ray.get(sum_task)
    logger.info(f"{total_sum = }")

    ray.shutdown()

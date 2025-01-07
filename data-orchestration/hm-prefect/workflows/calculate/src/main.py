import secrets
import time

from prefect import flow, get_run_logger, task
from pydantic import BaseModel


class Model(BaseModel):
    n: int


@task
def expand(n: int) -> list[int]:
    time.sleep(secrets.SystemRandom().uniform(0.5, 5))
    return [*range(n)]


@task
def power(a: int, b: int) -> int:
    time.sleep(secrets.SystemRandom().uniform(0.5, 2))
    return a**b


@task
def multiply(a: int, b: int) -> int:
    time.sleep(secrets.SystemRandom().uniform(0.5, 2))
    return a * b


@task
def add(a: int, b: int) -> int:
    time.sleep(secrets.SystemRandom().uniform(0.5, 2))
    return a + b


@task
def sum_up(nums: list[int]) -> int:
    time.sleep(secrets.SystemRandom().uniform(0.5, 2))
    return sum(nums)


@flow
def calculate(model: Model) -> None:
    logger = get_run_logger()
    nums = expand(model.n)
    nums = power.map(nums, 2)
    res = []
    for n in nums:
        ns = expand(n)
        ns = multiply.map(ns, 100)
        ns = add.map(ns, 9)
        res += ns
    n = sum_up(res)
    logger.info(n)


if __name__ == "__main__":
    external_model = Model(n=4)
    calculate(external_model)

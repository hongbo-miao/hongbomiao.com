from prefect import flow, get_run_logger, task


@task
def expand(n):
    return iter(range(n))


@task
def power(a, b):
    return a**b


@task
def multiply(a, b):
    return a * b


@task
def add(a, b):
    return a + b


@task
def print_nums(nums: list[int]) -> None:
    logger = get_run_logger()
    logger.info(nums)


@flow
def calculate(nums: list[int]) -> None:
    nums = power.map(nums, 2)
    for i in nums:
        x = expand(i)
        x = multiply.map(x, 100)
        x = add.map(x, 9)
        print_nums(x)


if __name__ == "__main__":
    calculate([*range(1, 4)])

from prefect import flow, get_run_logger, task


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
def minus(a, b):
    return a - b


@task
def print_nums(nums: list[int]) -> None:
    logger = get_run_logger()
    logger.info(nums)


@flow
def calculate(nums: list[int]) -> None:
    x = power.map(nums, 2)
    x = multiply.map(x, 100)
    x = add.map(x, 9)
    x = minus.map(x, 4)
    print_nums(x)


if __name__ == "__main__":
    calculate([*range(30)])

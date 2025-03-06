import daft
from prefect import flow, get_run_logger


@flow
def daft_analysis() -> None:
    logger = get_run_logger()
    df = daft.from_pydict(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [True, True, False, False, False],
        },
    )
    logger.info(f"{df = }")


if __name__ == "__main__":
    daft_analysis()

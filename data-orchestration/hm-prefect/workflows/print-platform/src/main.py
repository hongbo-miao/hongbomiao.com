import platform

from prefect import flow, get_run_logger


@flow
def hm_print_platform() -> None:
    logger = get_run_logger()
    logger.info(f"Node: {platform.node()}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python Version: {platform.python_version()}")


if __name__ == "__main__":
    hm_print_platform()

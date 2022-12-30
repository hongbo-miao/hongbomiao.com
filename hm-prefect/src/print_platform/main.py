import platform

from prefect import flow, get_run_logger


@flow
def print_platform():
    logger = get_run_logger()
    logger.info(f"node: {platform.node()}")
    logger.info(f"platform: {platform.platform()}")


if __name__ == "__main__":
    print_platform()

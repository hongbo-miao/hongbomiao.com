import logging

import matlab.engine

logger = logging.getLogger(__name__)


def main() -> None:
    matlab_engine = matlab.engine.start_matlab()
    greatest_common_divisor = matlab_engine.gcd(100.0, 80.0)
    logger.info(greatest_common_divisor)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()

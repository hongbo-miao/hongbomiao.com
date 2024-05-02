import logging

import matlab.engine


def main() -> None:
    matlab_engine = matlab.engine.start_matlab()
    greatest_common_divisor = matlab_engine.gcd(100.0, 80.0)
    logging.info(greatest_common_divisor)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

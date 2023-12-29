import logging
import sys

from pyspark.sql import SparkSession


def main() -> None:
    SparkSession.builder.getOrCreate()
    logging.info(sys.version_info)
    assert (sys.version_info.major, sys.version_info.minor) == (3, 11)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

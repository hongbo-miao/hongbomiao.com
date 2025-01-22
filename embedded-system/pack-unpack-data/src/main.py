import logging
import struct

logger = logging.getLogger(__name__)


def main() -> None:
    # https://docs.python.org/3/library/struct.html#byte-order-size-and-alignment
    # https://docs.python.org/3/library/struct.html#format-characters
    format_string = "<BLLcxx"
    size = struct.calcsize(format_string)
    logger.info(f"{size = }")

    values = (1, 1000, 2000, b"A")
    packed_data = struct.pack(format_string, *values)
    logger.info(f"{packed_data = }")

    unpacked_data = struct.unpack(format_string, packed_data)
    logger.info(f"{unpacked_data = }")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()

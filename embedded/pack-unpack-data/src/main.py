import logging
import struct


def main() -> None:
    # https://docs.python.org/3/library/struct.html#byte-order-size-and-alignment
    # https://docs.python.org/3/library/struct.html#format-characters
    format_string = "<BLLcxx"
    size = struct.calcsize(format_string)
    logging.info(f"{size = }")

    values = (1, 1000, 2000, b"A")
    packed_data = struct.pack(format_string, *values)
    logging.info(f"{packed_data = }")

    unpacked_data = struct.unpack(format_string, packed_data)
    logging.info(f"{unpacked_data = }")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

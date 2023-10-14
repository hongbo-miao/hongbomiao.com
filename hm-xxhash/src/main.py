import logging
from pathlib import Path

import xxhash


def get_file_xxh128(file_path: Path) -> str:
    hash = xxhash.xxh128()
    with open(file_path, "rb") as file:
        while True:
            data = file.read(8192)  # Read 8192 bytes at a time to use less memory
            if not data:
                break
            hash.update(data)
    return hash.hexdigest()


def main() -> None:
    file_path = Path("/path/to/file.txt")
    xxh128 = get_file_xxh128(file_path)
    logging.info(xxh128)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

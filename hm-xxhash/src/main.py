import logging
from pathlib import Path

import xxhash

logger = logging.getLogger(__name__)


def get_file_xxh128(file_path: Path) -> str:
    xxh128_hash = xxhash.xxh128()
    with open(file_path, "rb") as file:
        while True:
            data = file.read(8192)  # Read 8192 bytes at a time to use less memory
            if not data:
                break
            xxh128_hash.update(data)
    return xxh128_hash.hexdigest()


def main() -> None:
    file_path = Path("src/main.py")
    xxh128 = get_file_xxh128(file_path)
    logger.info(xxh128)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()

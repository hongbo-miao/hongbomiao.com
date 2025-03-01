import logging
from pathlib import Path

from utils.iads_util import IadsUtil

logger = logging.getLogger(__name__)


def main() -> None:
    iads_config_path = Path("pfConfig")
    IadsUtil.process_config(iads_config_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()

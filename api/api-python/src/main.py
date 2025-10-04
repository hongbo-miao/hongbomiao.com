import logging

import uvicorn
from config import config

logger = logging.getLogger(__name__)


def main() -> None:
    logger.info(f"Listening at {config.SERVER_HOST}:{config.SERVER_PORT}")
    uvicorn.run(
        "app:app",
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        reload=config.SERVER_RELOAD,
        log_level="info",
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()

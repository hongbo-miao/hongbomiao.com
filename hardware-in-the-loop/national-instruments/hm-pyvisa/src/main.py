import logging

import pyvisa

logger = logging.getLogger(__name__)


def main() -> None:
    resource_manager = pyvisa.ResourceManager()
    logger.info(resource_manager.list_resources())
    logger.info(resource_manager.list_opened_resources())

    instrument = resource_manager.open_resource("ASRL1::INSTR")
    logger.info(instrument)

    instrument.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()

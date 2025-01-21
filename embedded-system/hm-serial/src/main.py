import logging

import serial.tools.list_ports

logger = logging.getLogger(__name__)


def main() -> None:
    available_ports = serial.tools.list_ports.comports()

    if not available_ports:
        logger.info("No serial ports found.")
    else:
        for port in available_ports:
            logger.info(port.device)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()

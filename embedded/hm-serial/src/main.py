import logging

import serial.tools.list_ports


def main() -> None:
    available_ports = serial.tools.list_ports.comports()

    if not available_ports:
        logging.info("No serial ports found.")
    else:
        for port in available_ports:
            logging.info(port.device)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

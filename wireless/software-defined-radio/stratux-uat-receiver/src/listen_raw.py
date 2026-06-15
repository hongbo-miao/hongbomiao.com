import logging
import time

from configure_libusb import configure_libusb
from constants import FTDI_VENDOR_ID, STRATUX_BAUD_RATE, STRATUX_PRODUCT_ID
from register_stratux_ftdi_product import register_stratux_ftdi_product

logger = logging.getLogger(__name__)


def listen_raw() -> None:
    """Print every byte the radio emits, raw, with no decoding.

    The simplest possible check that the radio is alive: the instant it
    demodulates a 978 MHz UAT frame, the bytes show up here.
    """
    configure_libusb()
    from pyftdi.ftdi import Ftdi

    register_stratux_ftdi_product()
    ftdi = Ftdi()
    ftdi.open(vendor=FTDI_VENDOR_ID, product=STRATUX_PRODUCT_ID)
    ftdi.reset()
    ftdi.set_baudrate(STRATUX_BAUD_RATE)
    ftdi.set_line_property(8, 1, "N")
    ftdi.set_latency_timer(1)
    ftdi.purge_buffers()

    logger.info("Radio open at 2 Mbaud. Printing raw bytes as they arrive (Ctrl-C to stop).")
    byte_count: int = 0
    try:
        while True:
            chunk: bytes = ftdi.read_data(4096)
            if chunk:
                byte_count += len(chunk)
                logger.info(f"+{len(chunk)} bytes (total {byte_count}): {chunk!r}")
            else:
                time.sleep(0.01)
    except KeyboardInterrupt:
        logger.info(f"Stopped. Received {byte_count} bytes total.")
    finally:
        ftdi.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    listen_raw()

import logging
import time
from collections.abc import Iterator

from configure_libusb import configure_libusb
from constants import FTDI_VENDOR_ID, STRATUX_BAUD_RATE, STRATUX_PRODUCT_ID
from models import RawUatMessage
from parse_uat_line import parse_uat_line
from register_stratux_ftdi_product import register_stratux_ftdi_product

logger = logging.getLogger(__name__)

_HEARTBEAT_INTERVAL_SECOND: float = 5.0
_READ_CHUNK_SIZE: int = 4096


def read_uat_messages_ftdi(
    baud_rate: int = STRATUX_BAUD_RATE,
) -> Iterator[RawUatMessage]:
    """Stream parsed UAT messages straight from the radio's FTDI chip over libusb.

    This is the macOS path: the custom FTDI product id never receives a serial
    driver, so there is no ``/dev/cu.*`` node to open with pyserial.
    """
    configure_libusb()
    from pyftdi.ftdi import Ftdi

    register_stratux_ftdi_product()
    ftdi = Ftdi()
    ftdi.open(vendor=FTDI_VENDOR_ID, product=STRATUX_PRODUCT_ID)
    try:
        ftdi.reset()
        ftdi.set_baudrate(baud_rate)
        ftdi.set_line_property(8, 1, "N")
        ftdi.set_latency_timer(1)
        ftdi.purge_buffers()

        buffer: bytearray = bytearray()
        last_data_time: float = time.monotonic()
        while True:
            chunk: bytes = ftdi.read_data(_READ_CHUNK_SIZE)
            if not chunk:
                if time.monotonic() - last_data_time > _HEARTBEAT_INTERVAL_SECOND:
                    logger.info(
                        "Listening on the radio, no UAT frames received yet "
                        "(978 MHz traffic is sparse; try outdoors with a clear sky view)."
                    )
                    last_data_time = time.monotonic()
                time.sleep(0.01)
                continue

            last_data_time = time.monotonic()
            buffer.extend(chunk)
            while b"\n" in buffer:
                line_bytes, _, remainder = buffer.partition(b"\n")
                buffer = bytearray(remainder)
                message: RawUatMessage | None = parse_uat_line(
                    line_bytes.decode("ascii", errors="ignore")
                )
                if message is not None:
                    yield message
    finally:
        ftdi.close()

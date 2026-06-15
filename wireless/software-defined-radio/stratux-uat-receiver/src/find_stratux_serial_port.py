from serial.tools import list_ports

from constants import FTDI_VENDOR_ID, STRATUX_PRODUCT_ID

_FALLBACK_PORT_MARKER_LIST: tuple[str, ...] = ("usbserial", "wchusbserial", "usbmodem")


def find_stratux_serial_port() -> str | None:
    """Locate the Stratux UATRadio serial device.

    Prefers an exact FTDI vendor/product match, then falls back to any
    USB-serial style device so the demo still works behind a generic bridge.
    """
    port_info_list = list(list_ports.comports())
    for port_info in port_info_list:
        if port_info.vid == FTDI_VENDOR_ID and port_info.pid == STRATUX_PRODUCT_ID:
            return port_info.device

    for port_info in port_info_list:
        if any(marker in port_info.device for marker in _FALLBACK_PORT_MARKER_LIST):
            return port_info.device

    return None

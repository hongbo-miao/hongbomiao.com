from configure_libusb import configure_libusb
from register_stratux_ftdi_product import register_stratux_ftdi_product


def find_stratux_ftdi_device() -> bool:
    """Report whether the Stratux UATRadio is reachable as a raw FTDI device.

    Used on macOS, where the custom FTDI product id never gets a serial-port
    node and the demo must talk to the chip over libusb instead.
    """
    configure_libusb()
    from pyftdi.ftdi import Ftdi

    register_stratux_ftdi_product()
    return len(Ftdi.list_devices()) > 0

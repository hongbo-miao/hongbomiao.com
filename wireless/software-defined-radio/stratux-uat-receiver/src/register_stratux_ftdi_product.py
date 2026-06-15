from constants import FTDI_PRODUCT_NAME, FTDI_VENDOR_ID, STRATUX_PRODUCT_ID


def register_stratux_ftdi_product() -> None:
    """Teach pyftdi about the Stratux custom FTDI product id.

    Idempotent: pyftdi raises if a product id is registered twice, so a repeat
    call is treated as a no-op.
    """
    from pyftdi.ftdi import Ftdi

    try:
        Ftdi.add_custom_product(
            vid=FTDI_VENDOR_ID,
            pid=STRATUX_PRODUCT_ID,
            pidname=FTDI_PRODUCT_NAME,
        )
    except ValueError:
        pass

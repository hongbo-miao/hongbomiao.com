from typing import Final

# The Stratux UATRadio v1.0 enumerates as an FTDI USB-serial bridge and streams
# demodulated UAT frames at 2 Mbaud.
STRATUX_BAUD_RATE: Final[int] = 2_000_000
FTDI_VENDOR_ID: Final[int] = 0x0403
STRATUX_PRODUCT_ID: Final[int] = 0x7028

# macOS does not bind a serial driver to the radio's custom FTDI product id, so
# the demo talks to the FTDI chip directly over libusb under this product name.
FTDI_PRODUCT_NAME: Final[str] = "stratux"

# DO-282B UAT ADS-B payload sizes after the radio strips Reed-Solomon parity.
SHORT_FRAME_BYTE_COUNT: Final[int] = 18
LONG_FRAME_BYTE_COUNT: Final[int] = 34

# Latitude and longitude are sent as fractions of a full circle over 2^24 counts.
LATITUDE_LONGITUDE_COUNT: Final[float] = 16_777_216.0
FULL_CIRCLE_DEGREE: Final[float] = 360.0

# Pressure altitude is encoded in 25 ft steps offset 1000 ft below sea level.
ALTITUDE_STEP_FOOT: Final[int] = 25
ALTITUDE_OFFSET_FOOT: Final[int] = 1000

# Vertical rate is encoded in 64 ft/min steps.
VERTICAL_RATE_STEP_FOOT_PER_MINUTE: Final[int] = 64

# Supersonic air/ground states multiply the reported velocity by 4.
SUPERSONIC_VELOCITY_MULTIPLIER: Final[int] = 4

# Base-40 alphabet used to pack the 8-character call sign and emitter category.
BASE40_ALPHABET: Final[str] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ  .."

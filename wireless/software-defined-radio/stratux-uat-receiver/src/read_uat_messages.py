from collections.abc import Iterator

import serial

from constants import STRATUX_BAUD_RATE
from models import RawUatMessage
from parse_uat_line import parse_uat_line


def read_uat_messages(
    port: str,
    baud_rate: int = STRATUX_BAUD_RATE,
) -> Iterator[RawUatMessage]:
    """Open the radio's serial port and yield parsed UAT messages forever."""
    with serial.Serial(port, baud_rate, timeout=1) as serial_port:
        while True:
            raw_line: bytes = serial_port.readline()
            if not raw_line:
                continue
            message: RawUatMessage | None = parse_uat_line(
                raw_line.decode("ascii", errors="ignore")
            )
            if message is not None:
                yield message

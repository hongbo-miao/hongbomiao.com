import argparse
import logging
from collections.abc import Iterator

from constants import SHORT_FRAME_BYTE_COUNT
from decode_uat_downlink import decode_uat_downlink
from find_stratux_ftdi_device import find_stratux_ftdi_device
from find_stratux_serial_port import find_stratux_serial_port
from models import MessageDirection, RawUatMessage, UatDownlinkMessage
from read_uat_messages import read_uat_messages
from read_uat_messages_ftdi import read_uat_messages_ftdi
from simulate_uat_stream import simulate_uat_stream

logger = logging.getLogger(__name__)


def format_downlink_message(message: UatDownlinkMessage) -> str:
    call_sign: str = message.call_sign or "------"
    position: str = (
        f"{message.latitude_degree:9.4f},{message.longitude_degree:10.4f}"
        if message.latitude_degree is not None
        else "        -,         -"
    )
    altitude: str = (
        f"{message.altitude_foot:6d} ft" if message.altitude_foot is not None else "     - ft"
    )
    speed: str = (
        f"{message.ground_speed_knot:3d} kt" if message.ground_speed_knot is not None else "  - kt"
    )
    track: str = (
        f"{message.track_degree:3d} deg" if message.track_degree is not None else "  - deg"
    )
    return (
        f"[{message.icao_address}] {call_sign:8s} {position} {altitude} "
        f"{speed} {track} {message.air_ground_state.name}"
    )


def run(message_iterator: Iterator[RawUatMessage]) -> None:
    downlink_count: int = 0
    uplink_count: int = 0
    seen_address_set: set[str] = set()

    for message in message_iterator:
        if message.direction == MessageDirection.UPLINK:
            uplink_count += 1
            continue
        if len(message.payload) < SHORT_FRAME_BYTE_COUNT:
            continue

        downlink_count += 1
        decoded_message: UatDownlinkMessage = decode_uat_downlink(message.payload)
        seen_address_set.add(decoded_message.icao_address)
        logger.info(format_downlink_message(decoded_message))
        logger.info(
            f"  totals: {downlink_count} downlink, {uplink_count} uplink, "
            f"{len(seen_address_set)} unique aircraft"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode 978 MHz UAT ADS-B traffic from a Stratux UATRadio.",
    )
    parser.add_argument(
        "--port",
        type=str,
        default=None,
        help="Serial port of the radio (auto-detected when omitted).",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Replay synthetic frames instead of reading the radio.",
    )
    arguments = parser.parse_args()

    if arguments.simulate:
        logger.info("Running in simulate mode (no radio required).")
        run(simulate_uat_stream())
        return

    port: str | None = arguments.port or find_stratux_serial_port()
    message_iterator: Iterator[RawUatMessage]
    if port is not None:
        logger.info(f"Reading UAT frames from serial port {port}.")
        message_iterator = read_uat_messages(port)
    elif find_stratux_ftdi_device():
        logger.info(
            "No serial port for the radio (expected on macOS); reading directly "
            "from the Stratux FTDI device over libusb."
        )
        message_iterator = read_uat_messages_ftdi()
    else:
        logger.error(
            "No Stratux UATRadio found. Plug it in (check with `ioreg -p IOUSB | grep "
            "Stratux`), or run with --simulate to see the demo without hardware."
        )
        return

    try:
        run(message_iterator)
    except KeyboardInterrupt:
        logger.info("Stopped.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()

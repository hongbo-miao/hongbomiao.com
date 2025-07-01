import logging
import socket
import struct
import time

logger = logging.getLogger(__name__)

X_PLANE_IP: str = "127.0.0.1"
X_PLANE_PORT: int = 49000


def send_command(sock: socket.socket, dataref_str: str, value: float) -> None:
    """Construct and send a DREF command to X-Plane."""
    dataref: bytes = dataref_str.encode("utf-8")
    header: bytes = b"DREF\0"
    value_bytes: bytes = struct.pack("<f", value)
    dataref_padded: bytes = (dataref + b"\0").ljust(500, b"\0")
    packet: bytes = header + value_bytes + dataref_padded
    assert len(packet) == 509
    try:
        destination: tuple[str, int] = (X_PLANE_IP, X_PLANE_PORT)
        sock.sendto(packet, destination)
        logger.info(f"  -> Sent: Set '{dataref_str}' to {value}")
    except Exception:
        logger.exception("Error sending packet.")


def main() -> None:
    logger.info("--- X-Plane Auto-Takeoff ---")
    logger.info("!!! ENSURE YOUR AIRCRAFT IS AT THE START OF A RUNWAY !!!")
    for i in range(5, 0, -1):
        logger.info(f"Starting in {i}...")
        time.sleep(1)

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        # Step 1: Pre-takeoff configuration
        logger.info("\n[Phase 1] Configuring for takeoff...")
        # Brakes OFF
        send_command(sock, "sim/cockpit2/controls/parking_brake_ratio", 0.0)
        # Flaps to 25%
        send_command(sock, "sim/cockpit2/controls/flap_ratio", 0.25)
        # Full throttle
        send_command(
            sock,
            "sim/cockpit2/engine/actuators/throttle_ratio_all",
            1.0,
        )

        # Step 2: Takeoff roll (wait for speed to build)
        logger.info("\n[Phase 2] Takeoff roll... (20 seconds)")
        time.sleep(20)

        # Step 3: Rotation (pull back on the yoke)
        logger.info("\n[Phase 3] Rotating for liftoff...")
        # Pull back gently
        send_command(sock, "sim/joystick/yoke_pitch_ratio", -0.5)

        # Step 4: Initial climb and cleanup
        logger.info("\n[Phase 4] Positive rate of climb, cleaning up...")
        # Wait a few seconds after liftoff
        time.sleep(4)
        # Gear up
        logger.info("  -> Gear UP")
        send_command(sock, "sim/cockpit2/controls/gear_handle_down", 0.0)
        # Continue climbing
        time.sleep(6)
        # Reduce back pressure
        logger.info("  -> Easing yoke pressure")
        send_command(sock, "sim/joystick/yoke_pitch_ratio", -0.15)
        # Climb further
        time.sleep(15)
        # Flaps up
        logger.info("  -> Flaps UP")
        send_command(sock, "sim/cockpit2/controls/flap_handle_request_ratio", 0.0)

        # Step 5: Neutralize controls
        logger.info("\n[Phase 5] Stabilizing climb. You have control.")
        # Neutralize yoke
        send_command(sock, "sim/joystick/yoke_pitch_ratio", 0.0)

        logger.info("\n--- Auto-Takeoff Sequence Finished ---")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()

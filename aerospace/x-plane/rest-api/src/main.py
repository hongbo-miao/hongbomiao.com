import logging
import time

import httpx

logger = logging.getLogger(__name__)

X_PLANE_IP = "127.0.0.1"
X_PLANE_API_PORT = 8086
BASE_URL = f"http://{X_PLANE_IP}:{X_PLANE_API_PORT}/api/v2"


def get_dataref_id(
    httpx_client: httpx.Client,
    dataref_name: str,
    dataref_id_cache: dict[str, int],
) -> int | None:
    """Retrieve the numeric ID for a given dataref name using dataref_id_cache to avoid redundant lookups."""
    if dataref_name in dataref_id_cache:
        return dataref_id_cache[dataref_name]

    logger.info(f"Looking up ID for '{dataref_name}'...")

    try:
        response = httpx_client.get(
            f"{BASE_URL}/datarefs",
            params={"filter[name]": dataref_name},
        )
        response.raise_for_status()
        data = response.json()

        if data and data.get("data"):
            dataref_id = data["data"][0]["id"]
            dataref_id_cache[dataref_name] = dataref_id
            return dataref_id

        logger.error(f"Dataref '{dataref_name}' not found.")
    except httpx.RequestError:
        logger.exception(f"Error looking up dataref ID for '{dataref_name}'")
        return None
    else:
        return None


def send_command(
    httpx_client: httpx.Client,
    dataref_name: str,
    value: float,
    dataref_id_cache: dict[str, int],
) -> None:
    """Send a command by first getting the dataref's numeric ID, then sending a PATCH request."""
    dataref_id = get_dataref_id(httpx_client, dataref_name, dataref_id_cache)
    if dataref_id is None:
        return

    try:
        response = httpx_client.patch(
            f"{BASE_URL}/datarefs/{dataref_id}/value",
            json={"data": value},
        )
        response.raise_for_status()
        logger.info(f"Set '{dataref_name}' to {value}")
    except httpx.RequestError:
        logger.exception(f"Error sending command for '{dataref_name}'")


def main() -> None:
    logger.info("--- X-Plane Auto-Takeoff ---")
    logger.info("!!! ENSURE YOUR AIRCRAFT IS AT THE START OF A RUNWAY !!!")

    for i in range(5, 0, -1):
        logger.info(f"Starting in {i}...")
        time.sleep(1)

    dataref_id_cache: dict[str, int] = {}

    with httpx.Client() as httpx_client:
        # Step 1: Pre-takeoff configuration
        logger.info("\n[Phase 1] Configuring for takeoff...")
        # Brakes OFF
        send_command(
            httpx_client,
            "sim/cockpit2/controls/parking_brake_ratio",
            0.0,
            dataref_id_cache,
        )
        # Flaps to 25%
        send_command(
            httpx_client,
            "sim/cockpit2/controls/flap_ratio",
            0.25,
            dataref_id_cache,
        )
        # Full throttle
        send_command(
            httpx_client,
            "sim/cockpit2/engine/actuators/throttle_ratio_all",
            1.0,
            dataref_id_cache,
        )

        # Step 2: Takeoff roll (wait for speed to build)
        logger.info("\n[Phase 2] Takeoff roll... (20 seconds)")
        time.sleep(20)

        # Step 3: Rotation (pull back on the yoke)
        logger.info("\n[Phase 3] Rotating for liftoff...")
        # Pull back gently
        send_command(
            httpx_client,
            "sim/joystick/yoke_pitch_ratio",
            -0.5,
            dataref_id_cache,
        )

        # Step 4: Initial climb and cleanup
        logger.info("\n[Phase 4] Positive rate of climb, cleaning up...")
        # Wait a few seconds after liftoff
        time.sleep(4)
        # Gear up
        logger.info("  -> Gear UP")
        send_command(
            httpx_client,
            "sim/cockpit2/controls/gear_handle_down",
            0.0,
            dataref_id_cache,
        )
        # Continue climbing
        time.sleep(6)
        # Reduce back pressure
        logger.info("  -> Easing yoke pressure")
        send_command(
            httpx_client,
            "sim/joystick/yoke_pitch_ratio",
            -0.15,
            dataref_id_cache,
        )
        # Climb further
        time.sleep(15)

        # Flaps up
        logger.info("  -> Flaps UP")
        send_command(
            httpx_client,
            "sim/cockpit2/controls/flap_handle_request_ratio",
            0.0,
            dataref_id_cache,
        )

        # Step 5: Neutralize controls
        logger.info("\n[Phase 5] Stabilizing climb. You have control.")
        # Neutralize yoke
        send_command(
            httpx_client,
            "sim/joystick/yoke_pitch_ratio",
            0.0,
            dataref_id_cache,
        )

        logger.info("\n--- Auto-Takeoff Sequence Finished ---")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()

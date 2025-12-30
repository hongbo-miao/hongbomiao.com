import logging
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image
from shared.simulation.states.camera_state import camera_state

if TYPE_CHECKING:
    from omni.isaac.sensor import Camera

logger = logging.getLogger(__name__)


def get_camera_observation(
    camera: "Camera | None" = None,
    resolution: tuple[int, int] = (640, 480),
) -> Image.Image:
    """
    Get camera observation from the quadrotor's onboard camera.

    Args:
        camera: Camera sensor object (from create_camera)
        resolution: Camera resolution (width, height) for fallback

    Returns:
        PIL Image from the camera

    """
    if camera is not None:
        camera_state["instance"] = camera

    if camera_state["instance"] is None:
        logger.warning("No camera available, using placeholder image")
        rgb_array = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
        image = Image.fromarray(rgb_array, mode="RGB")
        logger.debug(f"Camera observation captured: {image.size}")
        return image

    try:
        rgb_data = camera_state["instance"].get_rgba()

        if rgb_data is None:
            logger.warning("Camera returned None, using placeholder image")
            rgb_array = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
        else:
            rgb_array = rgb_data[:, :, :3].astype(np.uint8)

    except (RuntimeError, AttributeError) as error:
        logger.warning(f"Camera capture failed: {error}")
        rgb_array = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)

    image = Image.fromarray(rgb_array, mode="RGB")

    logger.debug(f"Camera observation captured: {image.size}")

    return image

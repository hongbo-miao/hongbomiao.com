import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from omni.isaac.sensor import Camera

logger = logging.getLogger(__name__)


def create_camera(
    camera_prim_path: str = "/World/Camera",
    resolution: tuple[int, int] = (640, 480),
    position_offset: tuple[float, float, float] = (0.0, -5.0, 3.0),
    orientation: tuple[float, float, float, float] | None = None,
) -> "Camera":
    """
    Create a camera in the scene.

    Args:
        camera_prim_path: USD prim path for the camera
        resolution: Camera resolution (width, height)
        position_offset: Camera position (x, y, z) in world coordinates
        orientation: Camera orientation as quaternion (w, x, y, z). If None, looks forward.

    Returns:
        Camera sensor object

    """
    from omni.isaac.core import World  # noqa: PLC0415
    from omni.isaac.sensor import Camera  # noqa: PLC0415

    world = World.instance()

    if orientation is None:
        orientation = np.array([0.9239, 0.3827, 0.0, 0.0])
    else:
        orientation = np.array(orientation)

    camera = Camera(
        prim_path=camera_prim_path,
        position=np.array(position_offset),
        orientation=orientation,
        frequency=30,
        resolution=resolution,
    )

    world.scene.add(camera)

    camera.initialize()

    for _ in range(5):
        world.step(render=True)

    logger.info(f"Camera created at {camera_prim_path} with resolution {resolution}")

    return camera


def update_camera_position(
    camera: "Camera",
    target_position: tuple[float, float, float],
    offset: tuple[float, float, float] = (-3.0, -3.0, 2.0),
) -> None:
    """
    Update camera position to follow a target.

    Args:
        camera: Camera sensor object
        target_position: Target (x, y, z) to follow
        offset: Camera offset from target (behind, side, above)

    """
    import math  # noqa: PLC0415

    new_position = np.array(
        [
            target_position[0] + offset[0],
            target_position[1] + offset[1],
            target_position[2] + offset[2],
        ],
    )

    direction_x = target_position[0] - new_position[0]
    direction_y = target_position[1] - new_position[1]
    direction_z = target_position[2] - new_position[2]

    yaw = math.atan2(direction_y, direction_x)
    horizontal_distance = math.sqrt(direction_x**2 + direction_y**2)
    pitch = math.atan2(-direction_z, horizontal_distance)

    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(0)
    sr = math.sin(0)

    orientation = np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
    )

    camera.set_world_pose(position=new_position, orientation=orientation)

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from omni.isaac.core.prims import XFormPrim

logger = logging.getLogger(__name__)


def create_quadrotor(
    prim_path: str = "/World/Quadrotor",
    position: tuple[float, float, float] = (0.0, 0.0, 1.0),
    orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> "XFormPrim":
    """
    Create a quadrotorquadrotor in the Isaac Sim scene.

    Args:
        prim_path: USD prim path for the quadrotor
        position: Initial position (x, y, z) in meters
        orientation: Initial orientation as quaternion (w, x, y, z)

    Returns:
        XFormPrim representing the quadrotor

    """
    from omni.isaac.core import World  # noqa: PLC0415
    from omni.isaac.core.prims import XFormPrim  # noqa: PLC0415
    from omni.isaac.core.utils.nucleus import get_assets_root_path  # noqa: PLC0415
    from omni.isaac.core.utils.stage import add_reference_to_stage  # noqa: PLC0415

    assets_root_path = get_assets_root_path()

    quadrotor_usd_path = f"{assets_root_path}/Isaac/Robots/Quadrotor/quadrotor.usd"

    logger.info(f"Loading quadrotor from: {quadrotor_usd_path}")

    add_reference_to_stage(
        usd_path=quadrotor_usd_path,
        prim_path=prim_path,
    )

    world = World.instance()

    quadrotor = world.scene.add(
        XFormPrim(
            prim_path=prim_path,
            name="quadrotor",
            position=np.array(position),
            orientation=np.array(orientation),
        ),
    )

    logger.info(f"Quadrotor created at {prim_path}")

    return quadrotor

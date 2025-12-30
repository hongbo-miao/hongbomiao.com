import logging

from shared.types.vehicle_state import VehicleState

logger = logging.getLogger(__name__)


def get_vehicle_state(
    _quadrotor_prim_path: str = "/World/Quadrotor",
) -> VehicleState:
    """
    Get current state of the quadrotor from Isaac Sim.

    Args:
        _quadrotor_prim_path: USD prim path to the quadrotor (reserved for future use)

    Returns:
        Current VehicleState

    """
    from omni.isaac.core import World  # noqa: PLC0415
    from omni.isaac.core.utils.rotations import quat_to_euler_angles  # noqa: PLC0415

    world = World.instance()

    quadrotor = world.scene.get_object("quadrotor")

    position, orientation = quadrotor.get_world_pose()

    euler_angles = quat_to_euler_angles(orientation)

    current_time = world.current_time

    # XFormPrim doesn't have velocity methods, so we use zeros
    # In a real implementation, you would use RigidPrim or track velocity manually
    state = VehicleState(
        position_x=float(position[0]),
        position_y=float(position[1]),
        position_z=float(position[2]),
        velocity_x=0.0,
        velocity_y=0.0,
        velocity_z=0.0,
        roll=float(euler_angles[0]),
        pitch=float(euler_angles[1]),
        yaw=float(euler_angles[2]),
        angular_velocity_roll=0.0,
        angular_velocity_pitch=0.0,
        angular_velocity_yaw=0.0,
        timestamp=float(current_time) if current_time is not None else 0.0,
    )

    logger.debug(
        f"Vehicle state: pos=({state.position_x:.3f}, {state.position_y:.3f}, "
        f"{state.position_z:.3f}), yaw={state.yaw:.3f}",
    )

    return state

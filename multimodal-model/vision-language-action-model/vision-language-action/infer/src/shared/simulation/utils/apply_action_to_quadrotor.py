import logging

import numpy as np
from shared.types.vehicle_state import VehicleState
from vision_language_action_lib.types.action_output import ActionOutput

logger = logging.getLogger(__name__)


def apply_action_to_quadrotor(
    action: ActionOutput,
    current_state: VehicleState,
    _quadrotor_prim_path: str = "/World/Quadrotor",
    action_scale: float = 0.1,
) -> VehicleState:
    """
    Apply action to quadrotor and return new state.

    Args:
        action: ActionOutput from the policy (6-DoF deltas)
        current_state: Current vehicle state
        _quadrotor_prim_path: USD prim path to the quadrotor (reserved for future use)
        action_scale: Scaling factor for actions

    Returns:
        New VehicleState after applying action

    """
    from omni.isaac.core import World  # noqa: PLC0415
    from omni.isaac.core.utils.rotations import euler_angles_to_quat  # noqa: PLC0415

    world = World.instance()

    quadrotor = world.scene.get_object("quadrotor")

    new_position_x = current_state.position_x + action.delta_x_mps * action_scale
    new_position_y = current_state.position_y + action.delta_y_mps * action_scale
    new_position_z = current_state.position_z + action.delta_z_mps * action_scale

    new_roll = current_state.roll + action.delta_roll_radps * action_scale
    new_pitch = current_state.pitch + action.delta_pitch_radps * action_scale
    new_yaw = current_state.yaw + action.delta_yaw_radps * action_scale

    new_position = np.array([new_position_x, new_position_y, new_position_z])
    new_orientation_quat = euler_angles_to_quat(
        np.array([new_roll, new_pitch, new_yaw]),
    )

    quadrotor.set_world_pose(
        position=new_position,
        orientation=new_orientation_quat,
    )

    physics_delta_time_s = (
        world.get_physics_dt() if world.get_physics_dt() is not None else 0.01
    )

    new_state = VehicleState(
        position_x=new_position_x,
        position_y=new_position_y,
        position_z=new_position_z,
        velocity_x=0.0,
        velocity_y=0.0,
        velocity_z=0.0,
        roll=new_roll,
        pitch=new_pitch,
        yaw=new_yaw,
        angular_velocity_roll=0.0,
        angular_velocity_pitch=0.0,
        angular_velocity_yaw=0.0,
        timestamp=current_state.timestamp + physics_delta_time_s,
    )

    logger.debug(
        f"Applied action, new position: ({new_state.position_x:.3f}, "
        f"{new_state.position_y:.3f}, {new_state.position_z:.3f})",
    )

    return new_state

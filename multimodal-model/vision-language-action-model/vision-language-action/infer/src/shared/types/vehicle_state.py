from pydantic import BaseModel


class VehicleState(BaseModel):
    """Current state of the vehicle from sensors."""

    position_x: float  # meters
    position_y: float  # meters
    position_z: float  # meters
    velocity_x: float  # m/s
    velocity_y: float  # m/s
    velocity_z: float  # m/s
    roll: float  # radians
    pitch: float  # radians
    yaw: float  # radians
    angular_velocity_roll: float  # rad/s
    angular_velocity_pitch: float  # rad/s
    angular_velocity_yaw: float  # rad/s
    timestamp: float  # seconds

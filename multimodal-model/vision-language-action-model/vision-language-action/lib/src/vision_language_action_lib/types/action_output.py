from pydantic import BaseModel


class ActionOutput(BaseModel):
    """Raw action output from Flow Matching policy (6-DoF)."""

    delta_x: float
    delta_y: float
    delta_z: float
    delta_roll: float
    delta_pitch: float
    delta_yaw: float

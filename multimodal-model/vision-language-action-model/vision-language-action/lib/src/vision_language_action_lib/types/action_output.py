from pydantic import BaseModel


class ActionOutput(BaseModel):
    """Raw action output from Flow Matching policy (6-DoF) in physical units."""

    delta_x_mps: float
    delta_y_mps: float
    delta_z_mps: float
    delta_roll_radps: float
    delta_pitch_radps: float
    delta_yaw_radps: float

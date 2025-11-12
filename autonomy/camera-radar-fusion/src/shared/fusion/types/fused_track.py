import numpy as np
from config import config
from pydantic import BaseModel, ConfigDict, Field
from shared.camera.types.camera_detection import CameraDetection
from shared.radar.types.radar_detection import RadarDetection


class FusedTrack(BaseModel):
    """Represents a fused track combining camera and radar information."""

    camera_detection: CameraDetection = Field(description="Associated camera detection")
    radar_detection: RadarDetection = Field(description="Associated radar detection")
    fusion_confidence: float = Field(description="Combined confidence score (0-1)")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def bounding_box(self) -> np.ndarray:
        """Fused bounding box from camera detection."""
        return self.camera_detection.bounding_box

    @property
    def class_name(self) -> str:
        """Fused class name from camera detection."""
        return self.camera_detection.class_name

    @property
    def class_id(self) -> int:
        """Fused class ID from camera detection."""
        return self.camera_detection.class_id

    @property
    def position_3d(self) -> np.ndarray:
        """3D position from radar detection."""
        return self.radar_detection.position_3d

    @property
    def velocity(self) -> float:
        """Velocity from radar detection."""
        return self.radar_detection.velocity

    @property
    def distance(self) -> float:
        """Distance from radar detection."""
        return self.radar_detection.distance

    @property
    def image_coordinate_x(self) -> float:
        """Image x coordinate (center of bounding box)."""
        return self.camera_detection.center_x

    @property
    def image_coordinate_y(self) -> float:
        """Image y coordinate (center of bounding box)."""
        return self.camera_detection.center_y

    @property
    def is_moving(self) -> bool:
        """Movement classification based on velocity threshold."""
        return self.velocity > config.MOVEMENT_VELOCITY_THRESHOLD_MPS

    def __repr__(self) -> str:
        return (
            f"FusedTrack(class={self.class_name}, "
            f"distance={self.distance:.1f}m, "
            f"velocity={self.velocity:.1f}m/s, "
            f"confidence={self.fusion_confidence:.2f})"
        )

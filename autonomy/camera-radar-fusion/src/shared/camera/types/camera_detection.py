import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class CameraDetection(BaseModel):
    """Represents a single camera object detection."""

    bounding_box: np.ndarray = Field(
        description="[x1, y1, x2, y2] in pixel coordinates",
    )
    confidence: float = Field(description="Detection confidence score (0-1)")
    class_id: int = Field(description="Class ID from detector")
    class_name: str = Field(
        description="Human-readable class name (e.g., 'car', 'person')",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def center_x(self) -> float:
        """Calculate center x coordinate in pixel coordinates."""
        return (self.bounding_box[0] + self.bounding_box[2]) / 2

    @property
    def center_y(self) -> float:
        """Calculate center y coordinate in pixel coordinates."""
        return (self.bounding_box[1] + self.bounding_box[3]) / 2

    def __repr__(self) -> str:
        return (
            f"CameraDetection(class={self.class_name}, "
            f"confidence={self.confidence:.2f}, "
            f"center=({self.center_x:.0f}, {self.center_y:.0f}))"
        )

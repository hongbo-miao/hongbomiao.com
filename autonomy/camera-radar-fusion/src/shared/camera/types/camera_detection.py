import numpy as np


class CameraDetection:
    """Represents a single camera object detection."""

    def __init__(
        self,
        bounding_box: np.ndarray,
        confidence: float,
        class_id: int,
        class_name: str,
    ) -> None:
        """
        Initialize camera detection.

        Args:
            bounding_box: [x1, y1, x2, y2] in pixel coordinates
            confidence: Detection confidence score (0-1)
            class_id: Class ID from detector
            class_name: Human-readable class name (e.g., "car", "person")

        """
        self.bounding_box = bounding_box
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name

        # Calculate center point in pixel coordinates
        self.center_x = (bounding_box[0] + bounding_box[2]) / 2
        self.center_y = (bounding_box[1] + bounding_box[3]) / 2

    def __repr__(self) -> str:
        return (
            f"CameraDetection(class={self.class_name}, "
            f"confidence={self.confidence:.2f}, "
            f"center=({self.center_x:.0f}, {self.center_y:.0f}))"
        )

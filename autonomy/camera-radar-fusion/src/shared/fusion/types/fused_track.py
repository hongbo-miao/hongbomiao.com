from shared.camera.types.camera_detection import CameraDetection
from shared.radar.types.radar_detection import RadarDetection


class FusedTrack:
    """Represents a fused track combining camera and radar information."""

    def __init__(
        self,
        camera_detection: CameraDetection,
        radar_detection: RadarDetection,
        fusion_confidence: float,
    ) -> None:
        """
        Initialize fused track.

        Args:
            camera_detection: Associated camera detection
            radar_detection: Associated radar detection
            fusion_confidence: Combined confidence score (0-1)

        """
        self.camera_detection = camera_detection
        self.radar_detection = radar_detection
        self.fusion_confidence = fusion_confidence

        # Fused properties combining both sensors
        self.bounding_box = camera_detection.bounding_box
        self.class_name = camera_detection.class_name
        self.class_id = camera_detection.class_id

        # 3D position and velocity from radar
        self.position_3d = radar_detection.position_3d
        self.velocity = radar_detection.velocity
        self.distance = radar_detection.distance

        # Image coordinates (use center of bounding box)
        self.image_coordinate_x = camera_detection.center_x
        self.image_coordinate_y = camera_detection.center_y

        # Movement classification
        self.is_moving = self.velocity > 0.5  # threshold in m/s

    def __repr__(self) -> str:
        return (
            f"FusedTrack(class={self.class_name}, "
            f"distance={self.distance:.1f}m, "
            f"velocity={self.velocity:.1f}m/s, "
            f"confidence={self.fusion_confidence:.2f})"
        )

import numpy as np


class RadarDetection:
    """Represents a single radar detection."""

    def __init__(
        self,
        position_3d: np.ndarray,
        velocity: float,
        radar_cross_section: float,
        image_coordinate_x: float,
        image_coordinate_y: float,
    ) -> None:
        """
        Initialize radar detection.

        Args:
            position_3d: 3D position in radar frame [x, y, z]
            velocity: Radial velocity (m/s)
            radar_cross_section: Radar cross section (dBsm)
            image_coordinate_x: Projected x coordinate in image
            image_coordinate_y: Projected y coordinate in image

        """
        self.position_3d = position_3d
        self.velocity = velocity
        self.radar_cross_section = radar_cross_section
        self.image_coordinate_x = image_coordinate_x
        self.image_coordinate_y = image_coordinate_y
        self.distance = np.linalg.norm(position_3d)

    def __repr__(self) -> str:
        return (
            f"RadarDetection(distance={self.distance:.1f}m, "
            f"velocity={self.velocity:.1f}m/s, "
            f"image_pos=({self.image_coordinate_x:.0f}, {self.image_coordinate_y:.0f}))"
        )

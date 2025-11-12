import numpy as np
from shared.radar.types.radar_detection import RadarDetection


def create_radar_detection(
    position_3d: np.ndarray,
    velocity: float,
    radar_cross_section: float,
    image_coordinate_x: float,
    image_coordinate_y: float,
) -> RadarDetection:
    """
    Create a radar detection object.

    Args:
        position_3d: 3D position in radar frame [x, y, z]
        velocity: Radial velocity (m/s)
        radar_cross_section: Radar cross section (dBsm)
        image_coordinate_x: Projected x coordinate in image
        image_coordinate_y: Projected y coordinate in image

    Returns:
        RadarDetection object

    """
    return RadarDetection(
        position_3d=position_3d,
        velocity=velocity,
        radar_cross_section=radar_cross_section,
        image_coordinate_x=image_coordinate_x,
        image_coordinate_y=image_coordinate_y,
    )

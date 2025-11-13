import numpy as np
from shared.camera.types.camera_detection import CameraDetection
from shared.fusion.utils.is_radar_inside_bounding_box import (
    is_radar_inside_bounding_box,
)
from shared.radar.types.radar_detection import RadarDetection


def calculate_distance_to_bounding_box(
    radar_detection: RadarDetection,
    camera_detection: CameraDetection,
) -> float:
    """
    Calculate minimum distance from radar point to bounding box edge.

    Args:
        radar_detection: Radar detection with image coordinates
        camera_detection: Camera detection with bounding box

    Returns:
        Minimum distance in pixels (0 if inside box)

    """
    if is_radar_inside_bounding_box(radar_detection, camera_detection):
        return 0.0

    x1, y1, x2, y2 = camera_detection.bounding_box
    radar_x = radar_detection.image_coordinate_x
    radar_y = radar_detection.image_coordinate_y

    # Calculate closest point on bounding box to radar point
    closest_x = np.clip(radar_x, x1, x2)
    closest_y = np.clip(radar_y, y1, y2)

    # Euclidean distance to closest point
    distance = np.sqrt((radar_x - closest_x) ** 2 + (radar_y - closest_y) ** 2)
    return float(distance)

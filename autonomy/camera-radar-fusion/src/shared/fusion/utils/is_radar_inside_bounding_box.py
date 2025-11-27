from shared.camera.types.camera_detection import CameraDetection
from shared.radar.types.radar_detection import RadarDetection


def is_radar_inside_bounding_box(
    radar_detection: RadarDetection,
    camera_detection: CameraDetection,
) -> bool:
    """
    Check if radar detection falls inside camera bounding box.

    Args:
        radar_detection: Radar detection with image coordinates
        camera_detection: Camera detection with bounding box

    Returns:
        True if radar point is inside bounding box

    """
    x1, y1, x2, y2 = camera_detection.bounding_box

    return (
        x1 <= radar_detection.image_coordinate_x <= x2
        and y1 <= radar_detection.image_coordinate_y <= y2
    )

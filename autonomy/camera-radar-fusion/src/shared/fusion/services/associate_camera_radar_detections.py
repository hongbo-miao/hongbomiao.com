import logging

import numpy as np
from shared.camera.types.camera_detection import CameraDetection
from shared.fusion.utils.calculate_distance_to_bounding_box import (
    calculate_distance_to_bounding_box,
)
from shared.radar.types.radar_detection import RadarDetection

logger = logging.getLogger(__name__)


def associate_camera_radar_detections(
    camera_detections: list[CameraDetection],
    radar_detections: list[RadarDetection],
    distance_threshold_pixels: float = 100.0,
) -> list[tuple[int, int]]:
    """
    Associate camera and radar detections using spatial proximity.

    Uses greedy nearest-neighbor matching based on:
    1. Radar points inside camera bounding boxes (preferred)
    2. Minimum distance to bounding box edges

    Args:
        camera_detections: List of camera detections
        radar_detections: List of radar detections with image coordinates
        distance_threshold_pixels: Maximum distance for association

    Returns:
        List of matched pairs as (camera_index, radar_index) tuples

    """
    if not camera_detections or not radar_detections:
        return []

    # Build distance matrix
    distance_matrix = np.zeros((len(camera_detections), len(radar_detections)))

    for camera_index, camera_detection in enumerate(camera_detections):
        for radar_index, radar_detection in enumerate(radar_detections):
            distance = calculate_distance_to_bounding_box(
                radar_detection,
                camera_detection,
            )
            distance_matrix[camera_index, radar_index] = distance

    # Greedy matching: assign each radar to closest camera
    matched_pairs = []
    used_camera_indices = set()
    used_radar_indices = set()

    # Sort by distance (closest first)
    camera_indices, radar_indices = np.where(
        distance_matrix <= distance_threshold_pixels,
    )
    distances = distance_matrix[camera_indices, radar_indices]
    sorted_indices = np.argsort(distances)

    for index in sorted_indices:
        camera_index = camera_indices[index]
        radar_index = radar_indices[index]

        # Skip if already matched
        if camera_index in used_camera_indices or radar_index in used_radar_indices:
            continue

        matched_pairs.append((camera_index, radar_index))
        used_camera_indices.add(camera_index)
        used_radar_indices.add(radar_index)

    logger.debug(
        f"Associated {len(matched_pairs)} pairs from {len(camera_detections)} "
        f"camera and {len(radar_detections)} radar detections",
    )

    return matched_pairs

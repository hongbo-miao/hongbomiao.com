import logging

import numpy as np
from shared.camera.services.detect_objects_in_camera import detect_objects_in_camera
from shared.camera.types.camera_detection import CameraDetection
from shared.fusion.services.associate_camera_radar_detections import (
    associate_camera_radar_detections,
)
from shared.fusion.services.create_fused_track import create_fused_track
from shared.fusion.types.fused_track import FusedTrack
from shared.radar.services.create_radar_detection import create_radar_detection
from shared.radar.types.radar_detection import RadarDetection
from ultralytics import YOLO

logger = logging.getLogger(__name__)


def fuse_camera_radar(
    image: np.ndarray,
    radar_points_3d: np.ndarray,
    radar_velocities: np.ndarray,
    radar_cross_sections: np.ndarray,
    image_points: np.ndarray,
    yolo_model: YOLO,
) -> tuple[list[FusedTrack], list[CameraDetection], list[RadarDetection]]:
    """
    Perform camera-radar sensor fusion.

    This function:
    1. Detects objects in camera image using YOLO
    2. Creates radar detection objects with projected coordinates
    3. Associates camera and radar detections using spatial proximity
    4. Creates fused tracks combining information from both sensors

    Args:
        image: Camera image (BGR format)
        radar_points_3d: Nx3 array of radar 3D positions
        radar_velocities: N array of radar velocities
        radar_cross_sections: N array of radar cross sections
        image_points: Nx2 array of radar points projected to image
        yolo_model: Pre-loaded YOLO model for object detection

    Returns:
        Tuple of:
        - List of fused tracks (matched camera + radar)
        - List of unmatched camera detections
        - List of unmatched radar detections

    """
    # Step 1: Detect objects in camera image
    camera_detections = detect_objects_in_camera(
        image=image,
        model=yolo_model,
        confidence_threshold=0.5,
    )

    # Step 2: Create radar detection objects
    radar_detections = []
    for index in range(len(radar_points_3d)):
        radar_detection = create_radar_detection(
            position_3d=radar_points_3d[index],
            velocity=radar_velocities[index],
            radar_cross_section=radar_cross_sections[index],
            image_coordinate_x=image_points[index, 0],
            image_coordinate_y=image_points[index, 1],
        )
        radar_detections.append(radar_detection)

    # Step 3: Associate camera and radar detections
    matched_pairs = associate_camera_radar_detections(
        camera_detections=camera_detections,
        radar_detections=radar_detections,
        distance_threshold_pixels=100.0,
    )

    # Step 4: Create fused tracks
    fused_tracks = []
    matched_camera_set = set()
    matched_radar_set = set()

    for camera_detection, radar_detection in matched_pairs:
        track = create_fused_track(camera_detection, radar_detection)
        fused_tracks.append(track)

        matched_camera_set.add(id(camera_detection))
        matched_radar_set.add(id(radar_detection))

    # Separate unmatched detections
    unmatched_camera_detections = [
        detection
        for detection in camera_detections
        if id(detection) not in matched_camera_set
    ]

    unmatched_radar_detections = [
        detection
        for detection in radar_detections
        if id(detection) not in matched_radar_set
    ]

    logger.info(
        f"Fusion results: {len(fused_tracks)} fused tracks, "
        f"{len(unmatched_camera_detections)} camera-only, "
        f"{len(unmatched_radar_detections)} radar-only",
    )

    return fused_tracks, unmatched_camera_detections, unmatched_radar_detections

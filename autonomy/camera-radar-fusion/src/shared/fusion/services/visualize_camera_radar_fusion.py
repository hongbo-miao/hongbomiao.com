import logging
from pathlib import Path

import cv2
import numpy as np
from config import config
from nuscenes.nuscenes import NuScenes
from scipy.spatial.transform import Rotation
from shared.constants.colors import COLOR_BLACK, COLOR_BLUE, COLOR_GREEN, COLOR_YELLOW
from shared.fusion.services.fuse_camera_radar import fuse_camera_radar
from shared.fusion.utils.convert_nuscenes_quaternion_to_scipy import (
    convert_nuscenes_quaternion_to_scipy,
)
from shared.fusion.utils.project_radar_to_camera import project_radar_to_camera
from shared.radar.services.load_radar_data import load_radar_data
from ultralytics import YOLO

logger = logging.getLogger(__name__)


def visualize_camera_radar_fusion(
    nuscenes_instance: NuScenes,
    nuscenes_scene_index: int,
    visualization_frame_count: int,
) -> None:
    """
    Visualize camera-radar fusion for a nuScenes scene.

    Args:
        nuscenes_instance: NuScenes dataset instance
        nuscenes_scene_index: Index of scene to visualize (0-9 for mini dataset)
        visualization_frame_count: Maximum number of frames to process

    """
    # Load YOLO model for camera object detection
    logger.info("Loading YOLO model...")
    yolo_model = YOLO(Path(config.YOLO_MODEL_PATH))
    logger.info("YOLO model loaded")

    # Get scene
    scene = nuscenes_instance.scene[nuscenes_scene_index]
    logger.info(
        f"Processing scene: {scene['name']}, Description: {scene['description']}",
    )

    # Get first sample
    sample_token = scene["first_sample_token"]

    frame_count = 0
    while sample_token != "" and frame_count < visualization_frame_count:
        sample = nuscenes_instance.get("sample", sample_token)

        # Get front camera data
        camera_token = sample["data"]["CAM_FRONT"]
        camera_sample = nuscenes_instance.get("sample_data", camera_token)
        camera_path = nuscenes_instance.get_sample_data_path(camera_token)

        # Load camera image
        image = cv2.imread(str(camera_path))
        if image is None:
            logger.warning(f"Failed to load image: {camera_path}")
            sample_token = sample["next"]
            continue

        # Get camera calibration
        camera_calibration_token = camera_sample["calibrated_sensor_token"]
        camera_calibration = nuscenes_instance.get(
            "calibrated_sensor",
            camera_calibration_token,
        )
        camera_intrinsic = np.array(camera_calibration["camera_intrinsic"])

        # Get front radar data (RADAR_FRONT)
        radar_token = sample["data"]["RADAR_FRONT"]
        radar_sample = nuscenes_instance.get("sample_data", radar_token)

        # Load radar data
        radar_data = load_radar_data(nuscenes_instance, radar_token)

        # Get transformations
        radar_calibration_token = radar_sample["calibrated_sensor_token"]
        radar_calibration = nuscenes_instance.get(
            "calibrated_sensor",
            radar_calibration_token,
        )

        # Build transformation matrix from radar to camera
        # First: radar -> vehicle, then: vehicle -> camera
        radar_to_vehicle = np.eye(4)
        radar_quaternion_scipy = convert_nuscenes_quaternion_to_scipy(
            radar_calibration["rotation"],
        )
        radar_rotation = Rotation.from_quat(radar_quaternion_scipy)
        radar_to_vehicle[:3, :3] = radar_rotation.as_matrix()
        radar_to_vehicle[:3, 3] = np.array(radar_calibration["translation"])

        # Build camera to vehicle transformation (then invert to get vehicle to camera)
        camera_to_vehicle = np.eye(4)
        camera_quaternion_scipy = convert_nuscenes_quaternion_to_scipy(
            camera_calibration["rotation"],
        )
        camera_rotation = Rotation.from_quat(camera_quaternion_scipy)
        camera_to_vehicle[:3, :3] = camera_rotation.as_matrix()
        camera_to_vehicle[:3, 3] = np.array(camera_calibration["translation"])

        # Invert to get vehicle to camera
        vehicle_to_camera = np.linalg.inv(camera_to_vehicle)

        radar_to_camera = vehicle_to_camera @ radar_to_vehicle

        # Project radar points to camera
        radar_points_3d = radar_data[:, :3]
        radar_velocities = radar_data[:, 3]
        radar_cross_sections = radar_data[:, 4]

        image_points = project_radar_to_camera(
            radar_points_3d,
            camera_intrinsic,
            radar_to_camera,
        )

        # Perform sensor fusion
        fused_tracks, unmatched_camera, unmatched_radar = fuse_camera_radar(
            image=image,
            radar_points_3d=radar_points_3d,
            radar_velocities=radar_velocities,
            radar_cross_sections=radar_cross_sections,
            image_points=image_points,
            yolo_model=yolo_model,
        )

        # Visualize
        visualization = image.copy()

        # Draw fused tracks (green bounding boxes with radar info)
        for track in fused_tracks:
            x1, y1, x2, y2 = track.bounding_box.astype(int)

            # Bounding box for fused tracks
            cv2.rectangle(visualization, (x1, y1), (x2, y2), COLOR_GREEN, 2)

            # Label with class, distance, and velocity
            label = f"{track.class_name} {track.distance:.1f}m"
            if track.is_moving:
                label += f" {track.velocity:.1f}m/s"

            # Background for text
            text_size = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                1,
            )[0]
            cv2.rectangle(
                visualization,
                (x1, y1 - text_size[1] - 4),
                (x1 + text_size[0], y1),
                COLOR_GREEN,
                -1,
            )

            # Label text
            cv2.putText(
                visualization,
                label,
                (x1, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                COLOR_BLACK,
                1,
            )

            # Draw center point
            center_x = int(track.image_coordinate_x)
            center_y = int(track.image_coordinate_y)
            cv2.circle(visualization, (center_x, center_y), 5, COLOR_GREEN, -1)

        # Draw unmatched camera detections (blue bounding boxes)
        for camera_detection in unmatched_camera:
            x1, y1, x2, y2 = camera_detection.bounding_box.astype(int)

            # Blue bounding box for camera-only detections
            cv2.rectangle(visualization, (x1, y1), (x2, y2), COLOR_BLUE, 2)

            label = f"{camera_detection.class_name} (cam only)"

            # Label text
            cv2.putText(
                visualization,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                COLOR_BLUE,
                1,
            )

        # Draw unmatched radar detections (yellow points)
        for radar_detection in unmatched_radar:
            pixel_x = int(radar_detection.image_coordinate_x)
            pixel_y = int(radar_detection.image_coordinate_y)

            # Check if point is within image bounds
            if (
                0 <= pixel_x < visualization.shape[1]
                and 0 <= pixel_y < visualization.shape[0]
            ):
                # Yellow circle for radar-only detections
                radius = max(
                    3,
                    min(10, int(radar_detection.radar_cross_section / 10)),
                )
                cv2.circle(visualization, (pixel_x, pixel_y), radius, COLOR_YELLOW, -1)

                # Draw distance
                cv2.putText(
                    visualization,
                    f"{radar_detection.distance:.1f}m",
                    (pixel_x + 8, pixel_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    COLOR_YELLOW,
                    1,
                )

        # Add info overlay
        info_y = 30
        cv2.putText(
            visualization,
            f"Scene: {scene['name']} | Frame: {frame_count + 1}",
            (10, info_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        info_y += 30
        cv2.putText(
            visualization,
            f"Fused: {len(fused_tracks)} | Camera-only: {len(unmatched_camera)} | "
            f"Radar-only: {len(unmatched_radar)}",
            (10, info_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        # Legend
        legend_y = visualization.shape[0] - 80
        cv2.putText(
            visualization,
            "Green Box = Fused Track (Camera + Radar)",
            (10, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        legend_y += 20
        cv2.putText(
            visualization,
            "Blue Box = Camera Only | Yellow Point = Radar Only",
            (10, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # Show image
        cv2.imshow("Camera-Radar Fusion", visualization)

        key = cv2.waitKey(100) & 0xFF  # 10 FPS playback
        if key == ord("q"):
            break
        if key == ord(" "):
            # Pause on space
            cv2.waitKey(0)

        # Move to next sample
        sample_token = sample["next"]
        frame_count += 1

    cv2.destroyAllWindows()
    logger.info(f"Processed {frame_count} frames from scene {scene['name']}")

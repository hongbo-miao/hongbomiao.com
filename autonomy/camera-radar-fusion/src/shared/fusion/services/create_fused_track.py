import logging

from config import config
from shared.camera.types.camera_detection import CameraDetection
from shared.fusion.types.fused_track import FusedTrack
from shared.radar.types.radar_detection import RadarDetection

logger = logging.getLogger(__name__)


def create_fused_track(
    camera_detection: CameraDetection,
    radar_detection: RadarDetection,
) -> FusedTrack:
    """
    Create a fused track from associated camera and radar detections.

    The fusion combines:
    - Visual classification from camera (object type, bounding box)
    - 3D position and velocity from radar
    - Combined confidence score

    Args:
        camera_detection: Camera detection with visual features
        radar_detection: Radar detection with position and velocity

    Returns:
        FusedTrack combining information from both sensors

    """
    # Calculate combined confidence
    # Weight camera confidence more heavily as it provides classification
    fusion_confidence = (
        config.CAMERA_CONFIDENCE_WEIGHT * camera_detection.confidence
        + config.FUSION_BASE_CONFIDENCE
    )

    track = FusedTrack(
        camera_detection=camera_detection,
        radar_detection=radar_detection,
        fusion_confidence=fusion_confidence,
    )

    logger.debug(f"Created fused track: {track}")
    return track

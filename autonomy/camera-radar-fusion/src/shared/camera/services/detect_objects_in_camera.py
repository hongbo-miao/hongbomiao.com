import logging

import numpy as np
from shared.camera.types.camera_detection import CameraDetection
from ultralytics import YOLO

logger = logging.getLogger(__name__)


def detect_objects_in_camera(
    image: np.ndarray,
    model: YOLO,
    confidence_threshold: float = 0.5,
) -> list[CameraDetection]:
    """
    Detect objects in camera image using YOLO.

    Args:
        image: Input image (BGR format from OpenCV)
        model: Pre-loaded YOLO model instance
        confidence_threshold: Minimum confidence for detections

    Returns:
        List of camera detections with bounding boxes and metadata

    """
    # Run inference
    results = model(image, verbose=False)

    detections = []
    for result in results:
        boxes = result.boxes

        for index in range(len(boxes)):
            confidence = float(boxes.conf[index])

            # Filter by confidence
            if confidence < confidence_threshold:
                continue

            # Extract bounding box coordinates [x1, y1, x2, y2]
            bounding_box = boxes.xyxy[index].cpu().numpy()

            # Get class information
            class_id = int(boxes.cls[index])
            class_name = model.names[class_id]

            detection = CameraDetection(
                bounding_box=bounding_box,
                confidence=confidence,
                class_id=class_id,
                class_name=class_name,
            )
            detections.append(detection)

    logger.debug(f"Detected {len(detections)} objects in camera image")
    return detections

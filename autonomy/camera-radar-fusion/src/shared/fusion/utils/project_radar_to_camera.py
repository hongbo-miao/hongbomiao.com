import numpy as np


def project_radar_to_camera(
    radar_points: np.ndarray,
    camera_intrinsic: np.ndarray,
    radar_to_camera_transform: np.ndarray,
) -> np.ndarray:
    """
    Project 3D radar points to 2D camera image coordinates.

    Args:
        radar_points: Nx3 array of radar points in radar frame (x, y, z)
        camera_intrinsic: 3x3 camera intrinsic matrix
        radar_to_camera_transform: 4x4 transformation matrix from radar to camera

    Returns:
        Nx2 array of image coordinates (u, v), or empty if no valid points

    """
    if radar_points.shape[0] == 0:
        return np.empty((0, 2))

    # Convert to homogeneous coordinates
    radar_points_homogeneous = np.hstack(
        [radar_points, np.ones((radar_points.shape[0], 1))],
    )

    # Transform to camera frame
    camera_points = (radar_to_camera_transform @ radar_points_homogeneous.T).T

    # Filter points behind camera
    valid_mask = camera_points[:, 2] > 0
    if not np.any(valid_mask):
        return np.empty((0, 2))

    camera_points = camera_points[valid_mask]

    # Project to image plane
    image_points = (camera_intrinsic @ camera_points[:, :3].T).T

    # Normalize by depth
    image_points[:, 0] /= image_points[:, 2]
    image_points[:, 1] /= image_points[:, 2]

    return image_points[:, :2]

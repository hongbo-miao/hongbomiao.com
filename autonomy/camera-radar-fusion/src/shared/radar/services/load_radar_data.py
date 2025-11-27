import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud


def load_radar_data(nuscenes_instance: NuScenes, radar_token: str) -> np.ndarray:
    """
    Load radar point cloud data.

    Args:
        nuscenes_instance: NuScenes dataset instance
        radar_token: Token for radar sample data

    Returns:
        Nx5 array where each row is [x, y, z, velocity, radar_cross_section]

    """
    radar_file = nuscenes_instance.get_sample_data_path(radar_token)

    # Load radar point cloud
    radar_point_cloud = RadarPointCloud.from_file(str(radar_file))

    # radar_point_cloud.points is 18xN, we want x, y, z, velocity (compensated), radar cross section
    # indices: 0=x, 1=y, 2=z, 8=velocity_x_compensated, 9=velocity_y_compensated, 5=radar_cross_section
    points = radar_point_cloud.points.T  # Nx18

    # Calculate radial velocity
    velocity = np.sqrt(points[:, 8] ** 2 + points[:, 9] ** 2)

    # Combine into Nx5 array
    return np.column_stack(
        [points[:, 0], points[:, 1], points[:, 2], velocity, points[:, 5]],
    )

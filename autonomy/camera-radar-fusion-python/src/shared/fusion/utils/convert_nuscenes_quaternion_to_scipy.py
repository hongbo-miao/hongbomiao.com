def convert_nuscenes_quaternion_to_scipy(
    nuscenes_quaternion: list[float],
) -> list[float]:
    """
    Convert nuScenes quaternion format to SciPy format.

    Args:
        nuscenes_quaternion: Quaternion in nuScenes format [w, x, y, z]

    Returns:
        Quaternion in SciPy format [x, y, z, w]

    """
    return [
        nuscenes_quaternion[1],
        nuscenes_quaternion[2],
        nuscenes_quaternion[3],
        nuscenes_quaternion[0],
    ]

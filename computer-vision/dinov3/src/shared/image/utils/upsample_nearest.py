import numpy as np


def upsample_nearest(array: np.ndarray, patch_size: int) -> np.ndarray:
    """Nearest upsample for 2D or 3D arrays with last-dim channels."""
    if array.ndim == 2:
        return array.repeat(patch_size, 0).repeat(patch_size, 1)
    if array.ndim == 3:
        rows, columns, channels = array.shape
        array_upsampled = array.repeat(patch_size, 0).repeat(patch_size, 1)
        return array_upsampled.reshape(
            rows * patch_size,
            columns * patch_size,
            channels,
        )
    message = "upsample_nearest expects (rows,cols) or (rows,cols,channels)"
    raise ValueError(message)

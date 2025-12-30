import logging
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)


def save_camera_observation(
    image: Image.Image,
    output_directory: Path,
    step: int,
) -> Path:
    """
    Save camera observation image to disk.

    Args:
        image: PIL Image to save
        output_directory: Directory to save images
        step: Current step number for filename

    Returns:
        Path to saved image

    """
    output_directory.mkdir(parents=True, exist_ok=True)

    filename = f"frame_{step:05d}.png"
    output_path = output_directory / filename

    image.save(output_path)

    logger.debug(f"Saved camera observation: {output_path}")

    return output_path

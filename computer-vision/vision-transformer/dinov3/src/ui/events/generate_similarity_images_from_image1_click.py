import logging

import gradio as gr
from PIL import Image
from shared.image.utils.create_similarity_images import create_similarity_images

logger = logging.getLogger(__name__)


def generate_similarity_images_from_image1_click(
    image1: Image.Image | None,
    image2: Image.Image | None,
    event: gr.SelectData,
) -> tuple[Image.Image | None, Image.Image | None]:
    """
    Process click event on image1 and generate similarity visualizations.

    Args:
        image1: First image from Gradio
        image2: Second image from Gradio
        event: Gradio SelectData containing click coordinates

    Returns:
        Tuple of two images with similarity heatmaps

    """
    if image1 is None or image2 is None:
        return None, None

    x1, y1 = event.index[0], event.index[1]
    logger.info(f"Clicked at ({x1}, {y1})")

    return create_similarity_images(image1, image2, x1, y1)

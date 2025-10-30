import math

from PIL import Image


def pad_image_to_multiple(
    pil_image: Image.Image,
    multiple: int = 16,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Pad PIL image on right/bottom so (H,W) are multiples of `multiple`."""
    width, height = pil_image.size
    height_padded = int(math.ceil(height / multiple) * multiple)
    width_padded = int(math.ceil(width / multiple) * multiple)
    if (height_padded, width_padded) == (height, width):
        return pil_image, (0, 0, 0, 0)
    canvas = Image.new("RGB", (width_padded, height_padded), (0, 0, 0))
    canvas.paste(pil_image, (0, 0))
    return canvas, (0, 0, width_padded - width, height_padded - height)

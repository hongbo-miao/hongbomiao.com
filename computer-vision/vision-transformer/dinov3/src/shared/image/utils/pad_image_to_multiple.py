import math

from PIL import Image


def pad_image_to_multiple(
    pil_image: Image.Image,
    multiple: int = 16,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """
    Pad PIL image on right/bottom so (H, W) are multiples of patch size P.

    Math:
        Image I in R^(H x W x 3) is divided into P x P patches.
        To ensure integer patch counts, pad dimensions to multiples of P:
            H' = ceil(H / P) * P
            W' = ceil(W / P) * P

        After padding, patch counts become exact integers:
            N_row = floor(H' / P)
            N_col = floor(W' / P)
            N_patches = N_row * N_col

    Returns:
        Tuple of (padded_image, (left, top, right, bottom) padding amounts).

    """
    width, height = pil_image.size

    # H' = ceil(H / P) * P
    height_padded = int(math.ceil(height / multiple) * multiple)
    # W' = ceil(W / P) * P
    width_padded = int(math.ceil(width / multiple) * multiple)

    if (height_padded, width_padded) == (height, width):
        return pil_image, (0, 0, 0, 0)

    # Create black canvas and paste original image at top-left
    canvas = Image.new("RGB", (width_padded, height_padded), (0, 0, 0))
    canvas.paste(pil_image, (0, 0))
    return canvas, (0, 0, width_padded - width, height_padded - height)

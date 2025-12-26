import numpy as np
import torch
from PIL import Image
from shared.image.utils.pad_image_to_multiple import pad_image_to_multiple
from torchvision import transforms


def pad_and_normalize_image(
    pil_image: Image.Image,
    multiple: int = 16,
) -> tuple[dict[str, torch.Tensor], np.ndarray, tuple[int, int, int, int]]:
    """
    Pad image to multiple of P, then normalize for ViT input.

    Math:
        1. Pad image I in R^(H x W x 3) to I' in R^(H' x W' x 3)
            where H' = ceil(H / P) * P, W' = ceil(W / P) * P.

        2. ToTensor: Convert pixel values [0, 255] to [0, 1]
            x' = x / 255

        3. ImageNet normalization per channel c:
            x''_c = (x'_c - mu_c) / sigma_c

            where mu = [0.485, 0.456, 0.406] (RGB means)
                sigma = [0.229, 0.224, 0.225] (RGB standard deviations)

    Returns:
        Tuple of (pixel_values dict, display array, padding box).

    """
    # Input: PIL Image (W, H) with RGB channels
    # Pad to multiple of P: (H, W, 3) -> (H', W', 3)
    image_padded, pad_box = pad_image_to_multiple(pil_image, multiple=multiple)

    # ToTensor: (H', W', 3) -> (3, H', W'), values [0, 255] -> [0, 1]
    # Normalize: (x - mu) / sigma per channel
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ],
    )

    # Transform and add batch dimension: (3, H', W') -> (1, 3, H', W')
    pixel_tensor = transform(image_padded).unsqueeze(0)
    # Output: pixel_tensor shape (1, 3, H', W') for model input
    display_numpy = np.array(image_padded, dtype=np.uint8)
    return {"pixel_values": pixel_tensor}, display_numpy, pad_box

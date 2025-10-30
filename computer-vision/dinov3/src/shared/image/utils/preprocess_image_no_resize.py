import numpy as np
import torch
from PIL import Image
from shared.image.utils.pad_image_to_multiple import pad_image_to_multiple
from torchvision import transforms


def preprocess_image_no_resize(
    pil_image: Image.Image,
    multiple: int = 16,
) -> tuple[dict[str, torch.Tensor], np.ndarray, tuple[int, int, int, int]]:
    """Pad (right/bottom) -> ToTensor -> Normalize (ImageNet stats)."""
    image_padded, pad_box = pad_image_to_multiple(pil_image, multiple=multiple)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ],
    )
    pixel_tensor = transform(image_padded).unsqueeze(0)
    display_numpy = np.array(image_padded, dtype=np.uint8)
    return {"pixel_values": pixel_tensor}, display_numpy, pad_box

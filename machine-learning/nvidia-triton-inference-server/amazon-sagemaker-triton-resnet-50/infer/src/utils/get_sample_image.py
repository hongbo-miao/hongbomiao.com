from pathlib import Path

import numpy as np
from PIL import Image


def get_sample_image(image_path: Path) -> list[list[list[float]]]:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    image = (np.array(image).astype(np.float32) / 255) - np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32,
    ).reshape(1, 1, 3)
    image = image / np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    image = np.transpose(image, (2, 0, 1))
    return image.tolist()

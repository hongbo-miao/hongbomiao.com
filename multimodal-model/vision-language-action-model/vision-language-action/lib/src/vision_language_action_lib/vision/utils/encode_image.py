import logging

import torch
from PIL import Image
from torch import Tensor
from torchvision import transforms
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


def encode_image(
    image: Image.Image,
    model: PreTrainedModel,
    device: torch.device,
    image_size: int = 518,
) -> Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ],
    )

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        vision_features = model(image_tensor).last_hidden_state

    logger.debug(
        f"Vision features shape: {vision_features.shape}, dtype: {vision_features.dtype}",
    )
    return vision_features

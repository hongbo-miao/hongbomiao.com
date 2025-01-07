import logging
from pathlib import Path

import httpx
import torch
from torchvision import models

logger = logging.getLogger(__name__)


def download_labels():
    labels_url = (
        "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    )
    labels_path = Path("models/labels.txt")

    if not labels_path.exists():
        logger.info("Downloading labels...")
        with httpx.Client() as client:
            response = client.get(labels_url)
            labels_path.write_bytes(response.content)
        logger.info("Labels downloaded successfully")
    else:
        logger.info("Labels file already exists")


def download_resnet18():
    model_path = Path("models/resnet18.ot")

    if not model_path.exists():
        logger.info("Downloading ResNet18...")
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.eval()
        traced_model = torch.jit.trace(model, torch.randn(1, 3, 224, 224))
        traced_model.save(model_path)
        logger.info("Model downloaded and saved successfully")
    else:
        logger.info("Model file already exists")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    download_resnet18()
    download_labels()

import logging
from pathlib import Path

import httpx
import torch
import torchvision.models as models


def download_labels():
    labels_url = (
        "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    )
    labels_path = Path("models/labels.txt")

    if not labels_path.exists():
        logging.info("Downloading labels...")
        with httpx.Client() as client:
            response = client.get(labels_url)
            labels_path.write_bytes(response.content)
        logging.info("Labels downloaded successfully")
    else:
        logging.info("Labels file already exists")


def download_resnet18():
    model_path = Path("models/resnet18.ot")

    if not model_path.exists():
        logging.info("Downloading ResNet18...")
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.eval()
        traced_model = torch.jit.trace(model, torch.randn(1, 3, 224, 224))
        traced_model.save(model_path)
        logging.info("Model downloaded and saved successfully")
    else:
        logging.info("Model file already exists")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_resnet18()
    download_labels()

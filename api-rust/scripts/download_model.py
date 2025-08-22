import logging
from pathlib import Path

import httpx
import torch
import torch.onnx
from torchvision import models

logger = logging.getLogger(__name__)


def download_labels() -> None:
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


def download_resnet18() -> None:
    model_path = Path("models/resnet18.onnx")

    if not model_path.exists():
        logger.info("Downloading ResNet18...")
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.eval()

        # Create dummy input for ONNX export
        dummy_input = torch.randn(1, 3, 224, 224)

        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            model_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )
        logger.info("Model downloaded and exported to ONNX successfully")
    else:
        logger.info("Model file already exists")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    download_resnet18()
    download_labels()

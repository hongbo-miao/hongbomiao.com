import logging

import torch
from transformers import AutoModel, PreTrainedModel

logger = logging.getLogger(__name__)


def load_vision_model(
    model_id: str,
    device: torch.device | None = None,
) -> tuple[PreTrainedModel, torch.device]:
    if device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu",
        )

    logger.info(f"Loading vision model: {model_id}")
    logger.info(f"Device: {device}")

    model = AutoModel.from_pretrained(model_id)
    model = model.to(device)
    model.eval()

    logger.info("Vision model loaded successfully")
    return model, device

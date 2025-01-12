import logging
from pathlib import Path

import torch
from torch.nn import Module

logger = logging.getLogger(__name__)


def save_model(model: Module, model_dir_path: Path) -> None:
    logger.info("Save the model.")
    path = model_dir_path / Path("model.pth")
    torch.save(model.cpu().state_dict(), path)

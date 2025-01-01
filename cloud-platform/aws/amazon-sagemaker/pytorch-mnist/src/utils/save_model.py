import logging
import os

import torch
import torch.utils.data
import torch.utils.data.distributed

logger = logging.getLogger(__name__)


def save_model(model, model_dir):
    logger.info("Save the model.")
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)

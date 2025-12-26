from dataclasses import dataclass

import torch
from transformers import PreTrainedModel


@dataclass
class ModelCache:
    model: PreTrainedModel | None = None
    device: torch.device | None = None
    patch_size: int | None = None

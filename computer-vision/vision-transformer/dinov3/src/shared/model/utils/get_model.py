import logging

import torch
from shared.model.states.model_cache import ModelCache
from shared.model.utils.get_patch_size_from_model import get_patch_size_from_model
from transformers import AutoModel, PreTrainedModel

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
CACHE_STATE = ModelCache()


def get_model(
    cache_state: ModelCache | None = None,
    model_id: str = DEFAULT_MODEL_ID,
) -> tuple[PreTrainedModel, torch.device, int]:
    """Get or load the model, device, and patch size."""
    if cache_state is None:
        cache_state = CACHE_STATE
    if cache_state.model is None:
        cache_state.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu",
        )
        logger.info(f"Device: {cache_state.device}")
        logger.info(f"Loading model: {model_id}")
        cache_state.model = AutoModel.from_pretrained(model_id).to(cache_state.device)
        cache_state.model.eval()
        cache_state.patch_size = get_patch_size_from_model(cache_state.model, 16)
        logger.info(f"Using patch size: {cache_state.patch_size}")
    return cache_state.model, cache_state.device, cache_state.patch_size

import logging

import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


def load_language_model(
    model_id: str,
    device: torch.device | None = None,
    torch_dtype: torch.dtype | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizer, torch.device]:
    if device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu",
        )

    if torch_dtype is None:
        torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    logger.info(f"Loading embedding model: {model_id}")
    logger.info(f"Device: {device}, dtype: {torch_dtype}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model = model.to(device)
    model.eval()

    logger.info("Embedding model loaded successfully")
    return model, tokenizer, device

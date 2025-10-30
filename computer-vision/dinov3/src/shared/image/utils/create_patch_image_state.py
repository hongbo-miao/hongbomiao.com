import logging
from typing import TypedDict

import numpy as np
import torch
from PIL import Image
from shared.image.utils.preprocess_image_no_resize import preprocess_image_no_resize
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


class PatchImageState(TypedDict):
    pil: Image.Image
    patch_size: int
    display: np.ndarray
    pixel_values: torch.Tensor
    height: int
    width: int
    row_count: int
    column_count: int
    dimension: int
    patch_embeddings: np.ndarray
    embeddings_flat: np.ndarray
    embeddings_normalized: np.ndarray


def create_patch_image_state(
    pil_image: Image.Image,
    model: PreTrainedModel,
    device: torch.device,
    patch_size: int,
) -> PatchImageState:
    """Compute patch-level image embeddings and metadata for a single image."""
    inputs, display_numpy, _ = preprocess_image_no_resize(
        pil_image,
        multiple=patch_size,
    )
    pixel_values = inputs["pixel_values"].to(device)
    _, _, height, width = pixel_values.shape
    row_count = height // patch_size
    column_count = width // patch_size

    with torch.no_grad():
        output = model(pixel_values=pixel_values)
    hidden_states = output.last_hidden_state.squeeze(0).detach().cpu().numpy()

    token_count, dimension = hidden_states.shape
    patch_count = row_count * column_count
    special_token_count = token_count - patch_count
    if special_token_count < 1:
        message = (
            f"Token shape mismatch. token_count={token_count}, rows*cols={patch_count}, "
            f"height x width={height}x{width}, patch_size={patch_size}"
        )
        raise RuntimeError(
            message,
        )

    patch_embeddings = hidden_states[special_token_count:, :].reshape(
        row_count,
        column_count,
        dimension,
    )
    embeddings_flat = patch_embeddings.reshape(-1, dimension)
    embeddings_normalized = embeddings_flat / (
        np.linalg.norm(embeddings_flat, axis=1, keepdims=True) + 1e-8
    )

    return {
        "pil": pil_image,
        "patch_size": patch_size,
        "display": display_numpy,
        "pixel_values": pixel_values,
        "height": height,
        "width": width,
        "row_count": row_count,
        "column_count": column_count,
        "dimension": dimension,
        "patch_embeddings": patch_embeddings,
        "embeddings_flat": embeddings_flat,
        "embeddings_normalized": embeddings_normalized,
    }

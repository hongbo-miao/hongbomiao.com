import logging
from typing import TypedDict

import numpy as np
import torch
from PIL import Image
from shared.image.utils.pad_and_normalize_image import pad_and_normalize_image
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
    """
    Compute patch-level image embeddings and metadata for a single image.

    Math:
        Given image I in R^(H x W x 3), divide into P x P patches:
            N_row = floor(H / P)
            N_col = floor(W / P)
            N_patches = N_row * N_col

        ViT outputs hidden states:
            H = [h_CLS, h_1, h_2, ..., h_{N_patches}] in R^((1 + N_patches) x D)

            - h_CLS (index 0): CLS token that aggregates global image information,
                used for image-level classification tasks.
            - h_1 to h_{N_patches}: Patch embeddings with spatial locality.

        We exclude h_CLS and use only patch embeddings h_i because we need
        spatially localized features for pixel-level similarity matching.

        L2-normalize each embedding (epsilon = 1e-8):
            h_hat_i = h_i / (||h_i||_2 + epsilon)

    """
    inputs, display_numpy, _ = pad_and_normalize_image(
        pil_image,
        multiple=patch_size,
    )
    pixel_values = inputs["pixel_values"].to(device)
    _, _, height, width = pixel_values.shape

    # N_row = floor(H / P), N_col = floor(W / P)
    row_count = height // patch_size
    column_count = width // patch_size

    # Forward pass through ViT to get hidden states
    # Output shape: (1 + N_patches, D) where 1 is CLS token
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

    # Extract patch embeddings h_1 to h_{N_patches}, excluding h_CLS (index 0)
    # We need spatially localized features, not global image representation
    # Shape: (N_row, N_col, D)
    patch_embeddings = hidden_states[special_token_count:, :].reshape(
        row_count,
        column_count,
        dimension,
    )

    # Flatten to (N_patches, D) for similarity computation
    embeddings_flat = patch_embeddings.reshape(-1, dimension)

    # L2-normalize: h_hat_i = h_i / (||h_i||_2 + epsilon), epsilon = 1e-8
    # This enables cosine similarity via dot product: s_i = h_hat_i dot q_hat
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

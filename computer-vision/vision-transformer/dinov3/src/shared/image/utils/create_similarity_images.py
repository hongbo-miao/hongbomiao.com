import io
import logging

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from shared.image.utils.convert_pixel_coordinates_to_patch_index import (
    convert_pixel_coordinates_to_patch_index,
)
from shared.image.utils.create_patch_image_state import create_patch_image_state
from shared.image.utils.upsample_nearest import upsample_nearest
from shared.model.utils.get_model import get_model

logger = logging.getLogger(__name__)

OVERLAY_ALPHA = 0.6


def create_similarity_images(
    image1: Image.Image,
    image2: Image.Image,
    x1: int,
    y1: int,
) -> tuple[Image.Image, Image.Image]:
    """
    Create similarity visualization images for Gradio interface.

    Math:
        Given query patch embedding q at clicked position (x, y):
            1. Normalize query (epsilon = 1e-8):
                q_hat = q / (||q||_2 + epsilon)

            2. Cosine similarity with each patch i:
                s_i = h_hat_i dot q_hat

            3. Reshape to 2D map:
                S in R^(N_row x N_col)

            4. Min-max normalize for display (epsilon = 1e-8):
                s_tilde_i = (s_i - min(S)) / (max(S) - min(S) + epsilon)

    Args:
        image1: PIL Image for first image
        image2: PIL Image for second image
        x1: X coordinate on first image
        y1: Y coordinate on first image

    Returns:
        Tuple of two PIL Images (image1 with heatmap, image2 with heatmap)

    """
    model, device, patch_size = get_model()

    logger.info("Extracting features...")
    state1 = create_patch_image_state(image1, model, device, patch_size)
    state2 = create_patch_image_state(image2, model, device, patch_size)

    if state1["dimension"] != state2["dimension"]:
        msg = "Embedding dimensions differ - use the same model for both images."
        raise RuntimeError(msg)

    # Convert pixel (x1, y1) to patch index: index = row * N_col + col
    patch_index1 = convert_pixel_coordinates_to_patch_index(
        x1,
        y1,
        patch_size,
        state1["column_count"],
    )
    logger.info(f"Image 1: Selected patch index {patch_index1} at ({x1}, {y1})")

    # Get query embedding q and normalize: q_hat = q / (||q||_2 + epsilon)
    query1 = state1["embeddings_flat"][patch_index1]
    query1_normalized = query1 / (np.linalg.norm(query1) + 1e-8)

    # Cosine similarity via dot product: s_i = h_hat_i dot q_hat
    # Since both are L2-normalized, this equals cos(theta_i)
    # Shape: (N_patches,)
    cosine_similarity1_to_1 = state1["embeddings_normalized"] @ query1_normalized

    # Reshape to 2D map: S in R^(N_row x N_col)
    cosine_map1 = cosine_similarity1_to_1.reshape(
        state1["row_count"],
        state1["column_count"],
    )

    # Cross-image similarity: compare query from image1 to all patches in image2
    cosine_similarity1_to_2 = state2["embeddings_normalized"] @ query1_normalized
    cosine_map2 = cosine_similarity1_to_2.reshape(
        state2["row_count"],
        state2["column_count"],
    )

    colormap = plt.get_cmap("magma")

    # Min-max normalize for visualization: s_tilde_i = (s_i - min(S)) / (max(S) - min(S) + epsilon)
    display1 = (cosine_map1 - cosine_map1.min()) / (np.ptp(cosine_map1) + 1e-8)
    rgba1 = colormap(display1)
    # Upsample from patch resolution to pixel resolution using nearest neighbor
    rgba1_upsampled = upsample_nearest(
        rgba1,
        state1["patch_size"],
    )

    display2 = (cosine_map2 - cosine_map2.min()) / (np.ptp(cosine_map2) + 1e-8)
    rgba2 = colormap(display2)
    rgba2_upsampled = upsample_nearest(
        rgba2,
        state2["patch_size"],
    )

    figure1, axis1 = plt.subplots(1, 1, figsize=(8, 8))
    axis1.imshow(state1["display"])
    axis1.imshow(rgba1_upsampled, alpha=OVERLAY_ALPHA)
    axis1.plot(x1, y1, "r+", markersize=20, markeredgewidth=3)
    axis1.set_title(f"Image 1: Self-similarity at ({x1}, {y1})", fontsize=12)
    axis1.axis("off")

    figure1.tight_layout()
    buffer1 = io.BytesIO()
    figure1.savefig(buffer1, format="png", dpi=150, bbox_inches="tight")
    buffer1.seek(0)
    output_image1 = Image.open(buffer1)
    plt.close(figure1)

    figure2, axis2 = plt.subplots(1, 1, figsize=(8, 8))
    axis2.imshow(state2["display"])
    axis2.imshow(rgba2_upsampled, alpha=OVERLAY_ALPHA)
    axis2.set_title(f"Image 2: Cross-similarity from Image 1 ({x1}, {y1})", fontsize=12)
    axis2.axis("off")

    figure2.tight_layout()
    buffer2 = io.BytesIO()
    figure2.savefig(buffer2, format="png", dpi=150, bbox_inches="tight")
    buffer2.seek(0)
    output_image2 = Image.open(buffer2)
    plt.close(figure2)

    return output_image1, output_image2

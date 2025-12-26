import logging
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# DINOv2 model configuration
DINOV2_REPO = "facebookresearch/dinov2"
DINOV2_MODEL = "dinov2_vits14"  # ViT-Small with patch size 14
DINOV2_PATCH_SIZE = 14
DINOV2_IMAGE_SIZE = 518  # Must be divisible by patch size: 518 / 14 = 37
DINOV2_FEATURE_DIMENSION = 384  # Embedding dimension for ViT-Small

# AnyUp model configuration
ANYUP_REPO = "wimmerth/anyup"
ANYUP_MODEL = "anyup_multi_backbone"


# Load and normalize image to ImageNet statistics: I in R^(H x W x 3)
# Output tensor shape: (1, 3, H, W) where H = W = image_size
def load_image(
    image_path: Path,
    image_size: int = DINOV2_IMAGE_SIZE,
) -> tuple[torch.Tensor, Image.Image]:
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ],
    )
    # Add batch dimension: (3, H, W) -> (1, 3, H, W)
    # Batch=1 because this code processes one image at a time (PyTorch models expect batch dimension)
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor, image


# Load DINOv2 model from torch hub
def load_dinov2_model(device: torch.device) -> torch.nn.Module:
    dinov2 = torch.hub.load(DINOV2_REPO, DINOV2_MODEL)
    dinov2 = dinov2.to(device)
    dinov2.eval()
    return dinov2


# Extract low-resolution features from DINOv2
# Input: image_tensor shape (1, 3, 518, 518) = (batch, 3, H, W)
# Output: F_lr shape (1, 384, 37, 37) = (batch, C, h, w)
# For patch size P=14 and image size H=W=518: h = w = H/P = 518/14 = 37
# This is a 14x reduction in spatial resolution
def extract_dinov2_features(
    dinov2: torch.nn.Module,
    image_tensor: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        features = dinov2.forward_features(image_tensor)
        # Shape: (batch, num_patches, feature_dim) where num_patches = h * w
        patch_tokens = features["x_norm_patchtokens"]
        batch_size = patch_tokens.shape[0]
        feature_dimension = patch_tokens.shape[2]  # C = 384 for ViT-S
        assert feature_dimension == DINOV2_FEATURE_DIMENSION, (
            f"Feature dimension mismatch: expected {DINOV2_FEATURE_DIMENSION}, got {feature_dimension}"
        )
        grid_size = int(patch_tokens.shape[1] ** 0.5)  # h = w = sqrt(num_patches)
        # Reshape from (batch, h*w, C) to (batch, h, w, C)
        feature_map = patch_tokens.reshape(
            batch_size,
            grid_size,
            grid_size,
            feature_dimension,
        )
        # Permute to (batch, C, h, w) for standard feature map format
        return feature_map.permute(0, 3, 1, 2)


# Load AnyUp model from torch hub
def load_anyup_model(device: torch.device) -> torch.nn.Module:
    upsampler = torch.hub.load(
        ANYUP_REPO,
        ANYUP_MODEL,
        use_natten=False,
        device=device,
    )
    upsampler.eval()
    return upsampler


# Upsample features using image guidance: F_hr = AnyUp(I, F_lr)
# Input: I shape (1, 3, 518, 518) = (batch, 3, H, W), F_lr shape (1, 384, 37, 37) = (batch, C, h, w)
# Output: F_hr shape (1, 384, 518, 518) = (batch, C, H, W)
# Uses cross-attention where each high-res position attends to low-res positions:
#   F_hr(x, y) = sum_{i,j} alpha_{(x,y),(i,j)} * F_lr(i, j)
#   - (x, y): position in high-res output (H x W grid)
#   - (i, j): position in low-res input (h x w grid)
#   - alpha: attention weights computed from image features
def upsample_features_with_anyup(
    upsampler: torch.nn.Module,
    high_resolution_image: torch.Tensor,
    low_resolution_features: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        return upsampler(high_resolution_image, low_resolution_features)


# Visualize low-resolution vs high-resolution features
# Input shapes: (batch, C, h, w) and (batch, C, H, W) where batch=1 for single image
# For display: select first batch [0], then average across C channels with mean(dim=0)
def visualize_feature_comparison(
    low_resolution_features: torch.Tensor,
    high_resolution_features: torch.Tensor,
    original_image: Image.Image,
    output_path: Path,
) -> None:
    # low_resolution_features shape: (1, C, h, w) = (batch, C, h, w)
    # [0] selects first batch: (C, h, w)
    # mean(dim=0) averages across C channels: (h, w)
    low_resolution_visualization = low_resolution_features[0].mean(dim=0).cpu().numpy()
    # high_resolution_features shape: (1, C, H, W) = (batch, C, H, W)
    # [0] selects first batch: (C, H, W)
    # mean(dim=0) averages across C channels: (H, W)
    high_resolution_visualization = (
        high_resolution_features[0].mean(dim=0).cpu().numpy()
    )

    _, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    axes[1].imshow(low_resolution_visualization, cmap="viridis")
    axes[1].set_title(
        f"Low-Resolution Features ({low_resolution_features.shape[2]}x{low_resolution_features.shape[3]})",
    )
    axes[1].axis("off")
    axes[2].imshow(high_resolution_visualization, cmap="viridis")
    axes[2].set_title(
        f"Upsampled Features ({high_resolution_features.shape[2]}x{high_resolution_features.shape[3]})",
    )
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved visualization to {output_path}")


def main(image_path: Path, output_path: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load models once
    logger.info("Loading DINOv2 model")
    dinov2 = load_dinov2_model(device)

    logger.info("Loading AnyUp model")
    upsampler = load_anyup_model(device)

    logger.info("Loading image")
    image_tensor, original_image = load_image(image_path)
    image_tensor = image_tensor.to(device)

    logger.info("Extracting DINOv2 features")
    low_resolution_features = extract_dinov2_features(dinov2, image_tensor)
    logger.info(f"Low-resolution feature shape: {low_resolution_features.shape}")

    logger.info("Upsampling features with AnyUp")
    high_resolution_features = upsample_features_with_anyup(
        upsampler,
        image_tensor,
        low_resolution_features,
    )
    logger.info(f"High-resolution feature shape: {high_resolution_features.shape}")

    logger.info("Visualizing feature comparison")
    visualize_feature_comparison(
        low_resolution_features,
        high_resolution_features,
        original_image,
        output_path,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    data_directory_path = Path("data")
    external_image_path = data_directory_path / "image.jpg"
    external_output_path = data_directory_path / "output.png"
    main(external_image_path, external_output_path)

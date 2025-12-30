import torch
from torch import Tensor, nn


class VisionProjection(nn.Module):
    """Project vision features to language model embedding space."""

    def __init__(
        self,
        vision_dimension: int = 384,
        language_dimension: int = 3584,
        hidden_dimension: int = 1024,
    ) -> None:
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(vision_dimension, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, language_dimension),
            nn.LayerNorm(language_dimension),
        )

    def forward(self, vision_features: Tensor) -> Tensor:
        return self.projection(vision_features)


def create_vision_projection(
    vision_dimension: int = 384,
    language_dimension: int = 3584,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> VisionProjection:
    if device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu",
        )

    projection = VisionProjection(
        vision_dimension=vision_dimension,
        language_dimension=language_dimension,
    )
    return projection.to(device=device, dtype=dtype)

from typing import Any


def get_patch_size_from_model(model: Any, default: int = 16) -> int:  # noqa: ANN401
    """Try to get patch size from model config or attributes."""
    if hasattr(model, "config") and hasattr(model.config, "patch_size"):
        patch_size = model.config.patch_size
        if isinstance(patch_size, (tuple, list)):
            return patch_size[0]
        return patch_size
    if hasattr(model, "patch_size"):
        patch_size = model.patch_size
        if isinstance(patch_size, (tuple, list)):
            return patch_size[0]
        return patch_size
    return default

import io

import httpx
from PIL import Image


def load_image(path_or_url: str) -> Image.Image:
    """Download an image from URL or load from local path and return a PIL RGB image."""
    if path_or_url.lower().startswith(("http://", "https://")):
        with httpx.Client(follow_redirects=True, timeout=10.0) as httpx_client:
            response = httpx_client.get(path_or_url)
            response.raise_for_status()
            data = response.content
        return Image.open(io.BytesIO(data)).convert("RGB")
    return Image.open(path_or_url).convert("RGB")

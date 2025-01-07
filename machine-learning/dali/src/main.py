import logging
import urllib.request
from pathlib import Path

import torch
from nvidia.dali import fn, pipeline_def, types
from nvidia.dali.plugin.pytorch import DALIGenericIterator

logger = logging.getLogger(__name__)


def download_sample_images(data_path: Path) -> None:
    # Create main directory if it does not exist
    data_path.mkdir(parents=True, exist_ok=True)

    # Create a class subdirectory (e.g., "class0")
    class_dir_path = data_path / Path("class0")
    class_dir_path.mkdir(parents=True, exist_ok=True)

    # Sample image URLs
    image_urls: list[str] = [
        "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
    ]

    # Download images into the class subdirectory
    try:
        for i, url in enumerate(image_urls):
            filepath = class_dir_path / f"image_{i}.jpg"
            if not filepath.exists():
                logger.info(f"Downloading {url} to {filepath}")
                urllib.request.urlretrieve(url, str(filepath))
    except Exception:
        logger.exception(f"Error downloading {url}")


@pipeline_def(batch_size=2, num_threads=2, device_id=None)
def image_pipeline(data_path: Path):
    jpegs, labels = fn.readers.file(
        file_root=data_path,
        random_shuffle=True,
        initial_fill=2,
    )

    images = fn.decoders.image(jpegs, device="cpu")
    images = fn.resize(images, resize_x=224, resize_y=224)
    images = fn.crop_mirror_normalize(
        images,
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        dtype=types.FLOAT,
    )
    images = fn.transpose(images, perm=[2, 0, 1])

    return images, labels


def get_num_samples(data_path: Path) -> int:
    image_files = list(data_path.rglob("*.jpg"))
    return len(image_files)


def main() -> None:
    batch_size: int = 2
    num_threads: int = 2

    # Create data directory and download sample images
    data_path = Path("data")
    download_sample_images(data_path)

    # Get total number of samples
    num_samples = get_num_samples(data_path)
    if num_samples == 0:
        logger.exception("No images available in the directory.")
        return

    pipe = image_pipeline(
        data_path=data_path,
        batch_size=batch_size,
        num_threads=num_threads,
    )
    pipe.build()

    dali_iter = DALIGenericIterator(
        pipelines=[pipe],
        output_map=["data", "label"],
        reader_name="Reader",
        auto_reset=True,
    )

    logger.info("Pipeline created successfully!")
    logger.info(f"Ready to process images from {data_path}")

    try:
        for i, data in enumerate(dali_iter):
            images: torch.Tensor = data[0]["data"]
            labels: torch.Tensor = data[0]["label"]
            logger.info(f"Batch {i}: Image shape: {images.shape}, Labels: {labels}")
    except StopIteration:
        logger.info("Finished processing all images.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()

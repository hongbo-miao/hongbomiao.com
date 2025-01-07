import logging
from pathlib import Path
from random import sample

import lancedb
import pandas as pd
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from PIL import Image

logger = logging.getLogger(__name__)

EMBEDDINGS = (
    get_registry()
    .get("open-clip")
    .create(
        name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        batch_size=64,
        device="cpu",
    )
)


class Pets(LanceModel):
    vector: Vector(EMBEDDINGS.ndims()) = EMBEDDINGS.VectorField()  # type: ignore[valid-type]
    image_uri: str = EMBEDDINGS.SourceField()


def main():
    db = lancedb.connect("/tmp/lancedb")

    # Create or get the table
    if "pets" in db:
        logger.info("Using existing table")
        table = db["pets"]
    else:
        logger.info("Creating new table")
        table = db.create_table("pets", schema=Pets, mode="overwrite")
        # Use a sampling of images from the specified directory
        image_dir = Path("data/images")
        logger.info(f"Loading images from directory: {image_dir}")
        uris = [str(f) for f in image_dir.glob("*.jpg")]
        uris = sample(uris, 1000)
        logger.info(f"Processing {len(uris)} images")
        table.add(pd.DataFrame({"image_uri": uris}))

    # Query using text
    query_text = "black cat"
    logger.info(f"Performing text search with query: '{query_text}'")
    search_results = table.search(query_text).limit(3).to_pydantic(Pets)
    for idx, result in enumerate(search_results):
        logger.info(f"Text search result {idx + 1}: {result.image_uri}")

    # Query using an image
    query_image_path = Path("data/images/samoyed_27.jpg").expanduser()
    logger.info(f"Performing image search with query image: {query_image_path}")
    query_image = Image.open(query_image_path)
    search_results = table.search(query_image).limit(3).to_pydantic(Pets)
    for idx, result in enumerate(search_results):
        logger.info(f"Image search result {idx + 1}: {result.image_uri}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()

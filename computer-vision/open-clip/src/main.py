import logging
from pathlib import Path
from random import sample

import lancedb
import pandas as pd
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from PIL import Image

logging.basicConfig(level=logging.INFO)

EMBEDDINGS = (
    get_registry()
    .get("open-clip")
    .create(
        name="ViT-B-32", pretrained="laion2b_s34b_b79k", batch_size=64, device="cpu"
    )
)


class Pets(LanceModel):
    vector: Vector(EMBEDDINGS.ndims()) = EMBEDDINGS.VectorField()  # type: ignore
    image_uri: str = EMBEDDINGS.SourceField()


def main():
    db = lancedb.connect("/tmp/lancedb")

    # Create or get the table
    if "pets" in db:
        logging.info("Using existing table")
        table = db["pets"]
    else:
        logging.info("Creating new table")
        table = db.create_table("pets", schema=Pets, mode="overwrite")
        # Use a sampling of images from the specified directory
        image_dir = Path("data/images")
        logging.info(f"Loading images from directory: {image_dir}")
        uris = [str(f) for f in image_dir.glob("*.jpg")]
        uris = sample(uris, 1000)
        logging.info(f"Processing {len(uris)} images")
        table.add(pd.DataFrame({"image_uri": uris}))

    # Query using text
    query_text = "black cat"
    logging.info(f"Performing text search with query: '{query_text}'")
    search_results = table.search(query_text).limit(3).to_pydantic(Pets)
    for idx, result in enumerate(search_results):
        logging.info(f"Text search result {idx + 1}: {result.image_uri}")

    # Query using an image
    query_image_path = Path("data/images/samoyed_27.jpg").expanduser()
    logging.info(f"Performing image search with query image: {query_image_path}")
    query_image = Image.open(query_image_path)
    search_results = table.search(query_image).limit(3).to_pydantic(Pets)
    for idx, result in enumerate(search_results):
        logging.info(f"Image search result {idx + 1}: {result.image_uri}")


if __name__ == "__main__":
    main()

import logging

import lance
import numpy as np
from lance.vector import vec_to_table

logger = logging.getLogger(__name__)


def main() -> None:
    rng = np.random.default_rng()

    # Create sample vectors (minimum 5000 recommended for meaningful indexing)
    num_vectors = 5000  # Increased from 1000 to meet minimum recommendation
    vector_dim = 128  # Dimension of each vector (common for embeddings)
    vectors = rng.standard_normal((num_vectors, vector_dim))

    # Create some distinct vectors at the beginning for demonstration
    # Make the first vector have a clear pattern
    vectors[0] = np.array([1.0] * 32 + [2.0] * 32 + [3.0] * 32 + [4.0] * 32)
    # Make the second vector similar to the first but with some variation
    vectors[1] = vectors[0] + rng.standard_normal(vector_dim) * 0.1

    # Convert to Lance table
    vector_table = vec_to_table(vectors)

    # Save to Lance dataset
    uri = "/tmp/lancedb/vectors.lance"
    dataset = lance.write_dataset(vector_table, uri, mode="overwrite")
    logger.info(
        "Dataset saved to %s with %d vectors of dimension %d",
        uri,
        num_vectors,
        vector_dim,
    )

    # https://lancedb.github.io/lancedb/concepts/index_ivfpq/
    # Create an index for vector similarity search
    # IVF-PQ is a composite index that combines inverted file index (IVF) and product quantization (PQ)
    # - IVF divides the vector space into Voronoi cells using K-means clustering
    # - PQ reduces dimensionality by dividing vectors into sub-vectors and quantizing them
    dataset.create_index(
        "vector",
        index_type="IVF_PQ",
        # num_partitions: The number of partitions (Voronoi cells) in the IVF portion
        # - Controls how the vector space is divided
        # - Higher values increase query throughput but may reduce recall
        # - Should be chosen to target a particular number of vectors per partition
        # - For 5000 vectors, we use 64 partitions (~78 vectors per partition)
        num_partitions=64,
        # num_sub_vectors: The number of sub-vectors created during Product Quantization (PQ)
        # - Controls the compression level and search accuracy
        # - Chosen based on desired recall and vector dimensionality
        # - Trade-off: more sub-vectors = better compression but potentially lower accuracy
        num_sub_vectors=16,
    )
    logger.info("Created vector similarity index")

    # Read back the dataset
    dataset = lance.dataset(uri)

    # Perform vector similarity search
    query_vector = vectors[1]
    logger.info(
        "Performing similarity search for vector with pattern [1.0]*32 + [2.0]*32 + [3.0]*32 + [4.0]*32",
    )

    # Find 5 nearest neighbors
    k = 5

    # nprobes:
    #   The number of probes determines the distribution of vector space.
    #   While a higher number enhances search accuracy, it also results in slower performance.
    #   Typically, setting nprobes to cover 5 - 10% of the dataset proves effective in achieving high recall with minimal latency.
    #
    # refine_factor:
    #   Refine the results by reading extra elements and re-ranking them in memory.
    #   A higher number makes the search more accurate but also slower.
    results = dataset.to_table(
        prefilter=True,
        nearest={
            "column": "vector",
            "k": k,
            "q": query_vector,
            "nprobes": 500,
            "refine_factor": 10,
        },
    ).to_pandas()

    logger.info("Nearest neighbors (distances show similarity, lower = more similar):")
    for idx, row in results.iterrows():
        vector_preview = np.array(row["vector"])
        logger.info(
            f"Result {idx + 1}/{k}: Distance: {row['_distance']:.4f}, Vector preview: {vector_preview[:8]}...",
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()

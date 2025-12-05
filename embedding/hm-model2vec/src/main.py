import logging

from model2vec import StaticModel

logger = logging.getLogger(__name__)


def main() -> None:
    model = StaticModel.from_pretrained("minishlab/potion-retrieval-32M")

    sentences = [
        "It was the best of times, it was the worst of times.",
        "All happy families are alike; each unhappy family is unhappy in its own way.",
        "In a hole in the ground there lived a hobbit.",
    ]

    embeddings = model.encode(sentences)
    logger.info(f"Embeddings shape: {embeddings.shape}")

    token_embeddings = model.encode_as_sequence(sentences)
    logger.info(f"Token embeddings count: {len(token_embeddings)}")

    for i, sentence in enumerate(sentences):
        logger.info(f"Sentence: {sentence}")
        logger.info(f"Token embeddings shape: {token_embeddings[i].shape}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()

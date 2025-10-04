import logging

from config import config
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def load_embedding_model() -> SentenceTransformer:
    try:
        model = SentenceTransformer(config.EMBEDDING_MODEL)
        logger.info("Successfully loaded embedding model")
    except Exception:
        logger.exception("Failed to load model.")
    else:
        return model

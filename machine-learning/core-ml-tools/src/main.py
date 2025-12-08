import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import coremltools

logger = logging.getLogger(__name__)


def extract_dimensions(shape_attribute: Any) -> list[int] | None:  # noqa: ANN401
    if shape_attribute is None:
        return None
    source = getattr(shape_attribute, "dimensions", shape_attribute)
    values = [getattr(dimension, "size", dimension) for dimension in source]
    return values if values else None


def log_features(features: Iterable[Any], section_title: str) -> None:
    logger.info(section_title)
    for feature in features:
        feature_type = feature.type.WhichOneof("Type")
        details = getattr(feature.type, feature_type)
        dimensions = extract_dimensions(getattr(details, "shape", None))
        shape_details = f" shape={dimensions}" if dimensions else ""
        logger.info(f"  - {feature.name}: {feature_type}{shape_details}")


def main() -> None:
    model_path = Path("data/model.mlmodel")
    model = coremltools.models.MLModel(str(model_path))
    model_spec = model.get_spec()
    log_features(model_spec.description.input, "Inputs:")
    log_features(model_spec.description.output, "Outputs:")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()

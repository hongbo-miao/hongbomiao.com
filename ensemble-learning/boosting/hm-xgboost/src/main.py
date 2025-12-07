import logging
from pathlib import Path

import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

OBJECTIVE_NAME = "binary:logistic"
METRIC_NAME = "logloss"
MAX_DEPTH_COUNT = 3
LEARNING_RATE_VALUE = 0.1
ITERATION_COUNT = 100


def main() -> None:
    random_seed_number = 42
    artifact_directory_path = Path("output")
    ubjson_model_path = artifact_directory_path / "model.ubj"

    logger.info("Load breast cancer dataset")
    dataset = load_breast_cancer()
    feature_array = dataset.data
    label_array = dataset.target

    logger.info(
        f"Dataset has {feature_array.shape[0]} samples and {feature_array.shape[1]} features",
    )

    (
        feature_train_array,
        feature_test_array,
        label_train_array,
        label_test_array,
    ) = train_test_split(
        feature_array,
        label_array,
        test_size=0.2,
        random_state=random_seed_number,
        stratify=label_array,
    )

    train_data_matrix = xgb.DMatrix(feature_train_array, label=label_train_array)
    test_data_matrix = xgb.DMatrix(feature_test_array, label=label_test_array)

    parameter_dict: dict[str, float | int | str] = {
        "objective": OBJECTIVE_NAME,
        "eval_metric": METRIC_NAME,
        "max_depth": MAX_DEPTH_COUNT,
        "eta": LEARNING_RATE_VALUE,
        "seed": random_seed_number,
    }

    logger.info("Start training XGBoost model on breast cancer dataset")
    model = xgb.train(
        params=parameter_dict,
        dtrain=train_data_matrix,
        num_boost_round=ITERATION_COUNT,
    )

    logger.info("Start evaluating XGBoost model on breast cancer dataset")
    prediction_probability_array = model.predict(test_data_matrix)
    prediction_label_array = (prediction_probability_array > 0.5).astype(int)

    accuracy_value = float(
        (prediction_label_array == label_test_array).sum() / label_test_array.shape[0],
    )
    logger.info(f"Breast cancer test accuracy: {accuracy_value}")

    logger.info(f"Save model in UBJSON format to {ubjson_model_path}")
    model.save_model(ubjson_model_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()

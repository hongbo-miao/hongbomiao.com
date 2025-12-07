import logging
from pathlib import Path

from catboost import CatBoostClassifier, Pool
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

METRIC_NAME = "Logloss"
MAX_DEPTH_COUNT = 3
LEARNING_RATE_VALUE = 0.1
ITERATION_COUNT = 100


def main() -> None:
    random_seed_number = 42
    artifact_directory_path = Path("output")
    cbm_model_path = artifact_directory_path / "model.cbm"
    coreml_model_path = artifact_directory_path / "model.mlmodel"

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

    train_data_pool = Pool(data=feature_train_array, label=label_train_array)
    test_data_pool = Pool(data=feature_test_array, label=label_test_array)

    logger.info("Start training CatBoost model on breast cancer dataset")
    model = CatBoostClassifier(
        loss_function=METRIC_NAME,
        eval_metric=METRIC_NAME,
        depth=MAX_DEPTH_COUNT,
        learning_rate=LEARNING_RATE_VALUE,
        random_seed=random_seed_number,
        iterations=ITERATION_COUNT,
    )

    model.fit(train_data_pool, eval_set=test_data_pool)

    logger.info("Start evaluating CatBoost model on breast cancer dataset")
    prediction_probability_array = model.predict_proba(test_data_pool)[:, 1]
    prediction_label_array = (prediction_probability_array > 0.5).astype(int)

    accuracy_value = float(
        (prediction_label_array == label_test_array).sum() / label_test_array.shape[0],
    )
    logger.info(f"Breast cancer test accuracy: {accuracy_value}")

    logger.info(f"Save model in CBM format to {cbm_model_path}")
    model.save_model(cbm_model_path, format="cbm")
    logger.info(f"Save model in CoreML format to {coreml_model_path}")
    model.save_model(coreml_model_path, format="coreml")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()

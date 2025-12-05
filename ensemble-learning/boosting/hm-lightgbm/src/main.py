import logging

import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

OBJECTIVE_NAME = "binary"
METRIC_NAME = "binary_logloss"
MAX_DEPTH_COUNT = 3
LEARNING_RATE_VALUE = 0.1
ITERATION_COUNT = 100


def main() -> None:
    random_seed_number = 42

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

    parameter_dict: dict[str, float | int | str] = {
        "objective": OBJECTIVE_NAME,
        "metric": METRIC_NAME,
        "max_depth": MAX_DEPTH_COUNT,
        "learning_rate": LEARNING_RATE_VALUE,
        "seed": random_seed_number,
    }

    logger.info("Start training LightGBM model on breast cancer dataset")
    train_dataset = lgb.Dataset(feature_train_array, label=label_train_array)
    test_dataset = lgb.Dataset(
        feature_test_array,
        label=label_test_array,
        reference=train_dataset,
    )

    model = lgb.train(
        params=parameter_dict,
        train_set=train_dataset,
        num_boost_round=ITERATION_COUNT,
        valid_sets=[test_dataset],
        valid_names=["test"],
    )

    logger.info("Start evaluating LightGBM model on breast cancer dataset")
    prediction_probability_array = model.predict(feature_test_array)
    prediction_label_array = (prediction_probability_array > 0.5).astype(int)

    accuracy_value = float(
        (prediction_label_array == label_test_array).sum() / label_test_array.shape[0],
    )
    logger.info(f"Breast cancer test accuracy: {accuracy_value}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()

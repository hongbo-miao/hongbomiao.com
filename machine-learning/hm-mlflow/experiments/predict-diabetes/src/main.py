import logging

import config
import mlflow
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def main():
    mlflow.set_tracking_uri(
        f"https://{config.MLFLOW_TRACKING_SERVER_USER_NAME}:{config.MLFLOW_TRACKING_SERVER_PASSWORD}@{config.MLFLOW_TRACKING_SERVER_HOST}"
    )
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    mlflow.sklearn.autolog()

    diabetes = load_diabetes()
    x_train, x_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target)

    random_forest_regressor = RandomForestRegressor(
        n_estimators=100, max_depth=6, max_features=3
    )
    random_forest_regressor.fit(x_train, y_train)

    predictions = random_forest_regressor.predict(x_test)
    logging.info(f"{predictions = }")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
